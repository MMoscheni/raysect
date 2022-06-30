# cython: language_level=3

# Copyright (c) 2014-2021, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from raysect.optical           cimport new_point3d
from libc.math                 cimport floor, exp, fmin
from raysect.core.math.sampler cimport SphereSampler
from raysect.core.math.random  cimport uniform
cimport cython


cdef class VolumeIntegrator:
    """
    Base class for integrators in InhomogeneousVolumeEmitter materials.

    The deriving class must implement the integrate() method.
    """

    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        """
        Performs a customised integration of the emission through a volume emitter.

        This is a virtual method and must be implemented in a sub class.

        :param Spectrum spectrum: Spectrum measured so far along ray path. Add your emission
          to this spectrum, don't override it.
        :param World world: The world scene-graph.
        :param Ray ray: The ray being traced.
        :param Primitive primitive: The geometric primitive to which this material belongs
          (i.e. a cylinder or a mesh).
        :param InhomogeneousVolumeEmitter material: The material whose emission needs to be
          integrated.
        :param Point3D start_point: The start point for integration in world space.
        :param Point3D end_point: The end point for integration in world space.
        :param AffineMatrix3D world_to_primitive: Affine matrix defining the coordinate
          transform from world space to the primitive's local space.
        :param AffineMatrix3D primitive_to_world: Affine matrix defining the coordinate
          transform from the primitive's local space to world space.
        """

        raise NotImplementedError("Virtual method integrate() has not been implemented.")


cdef class NumericalIntegrator(VolumeIntegrator):
    """
    A basic implementation of the trapezium integration scheme for volume emitters.

    :param float step: The step size for numerical integration in metres.
    :param int min_samples: The minimum number of samples to use over integration
      range (default=5).
    """

    def __init__(self, float step, int min_samples = 5):
        self._step = step
        self._min_samples = min_samples

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, double value):
        if value <= 0:
            raise ValueError("Numerical integration step size can not be less than or equal to zero")
        self._step = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, int value):
        if value < 2:
            raise ValueError("At least two samples are required to perform the numerical integration.")
        self._min_samples = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef Spectrum integrate(self, Spectrum spectrum, World world, Ray ray, Primitive primitive,
                             InhomogeneousVolumeEmitter material, Point3D start_point, Point3D end_point,
                             AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        """
        Given origin and end points of a ray, source integration is performed along the linear trajectory
        (no scattering) connecting them. 

        By default, uniform sampling along the trajectory is assumed, and samples are uniformly spaced of
        a quantity step [m].

        If non-uniform sampling is activated, samples are distributed according to step_function_3d, i.e.
        step = step(x,y,z) and may vary from point to point. If step(x,y,z) == 0, step_max is used instead
        (source(x,y,z) == 0).

        If self-absorption is activated, the integrated emission between to consecutive sample points is
        linearly attenuated (Lambert-Beer law) of a factor exp(-step*absorption_function_3d(x,y,z)) where:
          - step is the step length (either uniform or non-uniform) [m]
          - absorption_function_3d(x,y,z) is the macroscopic self-absorption cross section [m^{-1}]
        """

        cdef:
          
            Point3D start, end
            Vector3D integration_direction, ray_direction
            double length, step, t, c
            double length_traveled = 0.0
            Spectrum emission, emission_previous, temp
            int intervals, interval, index
            
            # CAUTION.
            # - Scattering NOT YET IMPLEMENTED
            # - When implemented, would be QUALITATIVE only: reverse ray-tracing
            #   will somewhat "bias" trajectories (i.e. free path computed with
            #   cross-section at the "future" point in space...)
            #
            # int collisions = 0
            # int collisions_max = material.collisions_max
            # double sn
            # double collision_probability = 0.0
            # SphereSampler sphere_sampler

        # convert start and end points to local space
        start = start_point.transform(world_to_primitive)
        end = end_point.transform(world_to_primitive)

        # obtain local space ray direction and integration length
        integration_direction = start.vector_to(end)
        length = integration_direction.get_length()

        # nothing to contribute?
        if length == 0.0:
            return spectrum

        integration_direction = integration_direction.normalise()
        ray_direction = integration_direction.neg()

        # create working buffers
        emission = ray.new_spectrum()
        emission_previous = ray.new_spectrum()

        ####################################

        if material.use_step_function == 0:

            # usual procedure with uniform step and for loop

            # calculate number of complete intervals (samples - 1)
            intervals = max(self._min_samples - 1, <int> floor(length / self._step))

            # adjust (increase) step size to absorb any remainder and maintain equal interval spacing
            step = length / intervals

            # sample point and sanity check as bounds checking is disabled
            emission_previous = material.emission_function(start, ray_direction, emission_previous, world, ray, primitive, world_to_primitive, primitive_to_world)
            self._check_dimensions(emission_previous, spectrum.bins)

            # numerical integration
            c = 0.5 * step
            for interval in range(0, intervals):

                # calculate location of sample point at the top of the interval
                t = (interval + 1) * step
                sample_point = new_point3d(
                    start.x + t * integration_direction.x,
                    start.y + t * integration_direction.y,
                    start.z + t * integration_direction.z
                )

                # sample point and sanity check as bounds checking is disabled
                emission = material.emission_function(sample_point, ray_direction, emission, world, ray, primitive, world_to_primitive, primitive_to_world)
                self._check_dimensions(emission, spectrum.bins)

                # trapezium rule integration
                for index in range(spectrum.bins):
                    spectrum.samples_mv[index] += c * (emission.samples_mv[index] + emission_previous.samples_mv[index])
                    if material.use_absorption_function == 1:
                        # linear attenuation if self-absorption is activated
                        spectrum.samples_mv[index] *= exp(- step * material.absorption_function_3d(sample_point.x, sample_point.y, sample_point.z))

                # swap buffers and clear the active buffer
                temp = emission_previous
                emission_previous = emission
                emission = temp
                emission.clear()

        ####################################

        elif material.use_step_function == 1 and material.use_scattering_function == 0:

            # new procedure with non-uniform step and while loop

            length_traveled = 0.0
            
            # sample point 
            emission_previous = material.emission_function(start, ray_direction, emission_previous,
                                                           world, ray, primitive,
                                                           world_to_primitive, primitive_to_world)

            # numerical integration
            
            while length_traveled < length:

                # calculate location of sample point at the top of the interval
                step = material.step_function_3d(start.x,
                                                 start.y,
                                                 start.z)
                
                # means that emission is exactly 0.0
                # and prevents infinite loop
                if step == 0.0: step = material.step_max
                
                start = new_point3d(
                    start.x + step * integration_direction.x,
                    start.y + step * integration_direction.y,
                    start.z + step * integration_direction.z
                )

                emission = material.emission_function(start, ray_direction, emission_previous,
                                                      world, ray, primitive,
                                                      world_to_primitive, primitive_to_world)

                for index in range(spectrum.bins):
                    spectrum.samples_mv[index] += 0.5 * step * (emission.samples_mv[index] + emission_previous.samples_mv[index])
                    if material.use_absorption_function == 1:
                        # linear attenuation if self-absorption is activated
                        spectrum.samples_mv[index] *= exp(- step * material.absorption_function_3d(sample_point.x, sample_point.y, sample_point.z))

                emission_previous = emission
                emission.clear()

                length_traveled += step
                
        ####################################
        
        # you need non-uniform sampling step to allow for scattering because of variable mfp

        # elif material.use_step_function == 1 and material.use_scattering_function == 1:
            
        #     # sample point 
        #     emission_previous = material.emission_function(start, ray_direction, emission_previous,
        #                                                    world, ray, primitive,
        #                                                    world_to_primitive, primitive_to_world)

        #     # numerical integration:
        #     # stops when either collisions_max reached OR primitive boundary reached
            
        #     while collisions <= collisions_max:

        #         # calculate smart sampling step
        #         step = material.step_function_3d(start.x,
        #                                          start.y,
        #                                          start.z)
                
        #         # macroscopic cross section: sigma * density [m^{-1}]
        #         # reciprocal = mean free path between two collisions [m]
        #         sn = material.scattering_function_3d(start.x,
        #                                              start.y,
        #                                              start.z)
                
        #         # means that emission is exactly 0.0 => step_max adopted
        #         if step == 0.0:
        #             step = material.step_max
                
        #         # minimum between step and mfp to properly resolve both emission and scattering, respectively
        #         # (HP. possibly scattering where step == 0, i.e. scattering > 0 although emission == 0)
        #         step = fmin(step, 1.0 / sn)
                  
        #         start = new_point3d(
        #             start.x + step * integration_direction.x,
        #             start.y + step * integration_direction.y,
        #             start.z + step * integration_direction.z
        #         )
                
        #         # check boundary has NOT been reached
        #         if primitive.contains(start) == True:

        #           emission = material.emission_function(start, ray_direction, emission_previous,
        #                                                 world, ray, primitive,
        #                                                 world_to_primitive, primitive_to_world)

        #           # trapezium rule integration
        #           # HP: no wavelength dependence
        #           spectrum.samples_mv[0] += 0.5 * step * (emission.samples_mv[0] + emission_previous.samples_mv[0])

        #           emission_previous = emission
        #           emission.clear()

        #           # what happens next?
        #           collision_probability = 1.0 - exp(- step * sn)

        #           # YES: collision condition is met
        #           # (strict "<" sign - no "=" - because NO collision if collision_probability == 0)
        #           if uniform() < collision_probability:

        #             # isotropic scattering
        #             sphere_sampler = SphereSampler()
        #             # sphere_sampler(1) return a list of 1 Vector3D and we take the 0th element
        #             integration_direction = sphere_sampler(1)[0]
        #             collisions += 1
                    
        #         else:
        #           # story ends
        #           return spectrum


        return spectrum

    cdef int _check_dimensions(self, Spectrum spectrum, int bins) except -1:
        if spectrum.samples.ndim != 1 or spectrum.samples.shape[0] != bins:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")


cdef class InhomogeneousVolumeEmitter(NullSurface):
    """
    Base class for inhomogeneous volume emitters.

    The integration technique can be changed by the user, but defaults to
    a basic numerical integration scheme.

    The deriving class must implement the emission_function() method.

    :param VolumeIntegrator integrator: Integration object, defaults to
      NumericalIntegrator(step=0.01, min_samples=5).
    """

    def __init__(self, VolumeIntegrator integrator=None):
        super().__init__()
        self.integrator = integrator or NumericalIntegrator(step=0.01, min_samples=5)
        self.importance = 1.0

    cpdef Spectrum evaluate_volume(self, Spectrum spectrum, World world,
                                   Ray ray, Primitive primitive,
                                   Point3D start_point, Point3D end_point,
                                   AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):

        # pass to volume integrator class
        return self.integrator.integrate(spectrum, world, ray, primitive, self, start_point, end_point,
                                         world_to_primitive, primitive_to_world)

    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world):
        """
        The emission function for the material at a given sample point.

        This is a virtual method and must be implemented in a sub class.

        :param Point3D point: Requested sample point in local coordinates.
        :param Vector3D direction: The emission direction in local coordinates.
        :param Spectrum spectrum: Spectrum measured so far along ray path. Add your emission
          to this spectrum, don't override it.
        :param World world: The world scene-graph.
        :param Ray ray: The ray being traced.
        :param Primitive primitive: The geometric primitive to which this material belongs
          (i.e. a cylinder or a mesh).
        :param AffineMatrix3D world_to_primitive: Affine matrix defining the coordinate
          transform from world space to the primitive's local space.
        :param AffineMatrix3D primitive_to_world: Affine matrix defining the coordinate
          transform from the primitive's local space to world space.
        """

        raise NotImplementedError("Virtual method emission_function() has not been implemented.")
        
