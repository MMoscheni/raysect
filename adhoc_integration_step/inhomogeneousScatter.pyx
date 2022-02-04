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
from libc.math                 cimport floor, fmin, exp, log
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
        cdef:
            
            Point3D start, end
            Vector3D integration_direction, ray_direction
            double length, step, t, c
            double length_traveled = 0.0
            Spectrum emission, emission_previous, temp
            int intervals, interval, index
            
            # scattering-absorption
            int collisions = 0
            int collisions_max = material.collisions_max
            double sn, step_sampling, fp
            # double collision_probability = 0.0
            SphereSampler sphere_sampler
            
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
        ############### MMM ################
        ####################################
        
        if material.use_step_function == 0:
            
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
                    # if material.use_absorption_function == 1:
                        # spectrum.samples_mv[index] *= exp(- step * material.absorption_function_3d(sample_point.x, sample_point.y, sample_point.z))
                
                # swap buffers and clear the active buffer
                temp = emission_previous
                emission_previous = emission
                emission = temp
                emission.clear()
                
        ####################################
        ############### MMM ################
        ####################################
        
        elif material.use_step_function == 1 and material.use_scattering_function == 0:
            
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
                if step == 0.0:
                    step = material.step_max
                
                start = new_point3d(
                    start.x + step * integration_direction.x,
                    start.y + step * integration_direction.y,
                    start.z + step * integration_direction.z
                )
                
                emission = material.emission_function(start, ray_direction, emission_previous,
                                                      world, ray, primitive,
                                                      world_to_primitive, primitive_to_world)
                # trapezium rule integration
                # HP: no wavelength dependence
                spectrum.samples_mv[0] += 0.5 * step * (emission.samples_mv[0] + emission_previous.samples_mv[0])
                # if material.use_absorption_function == 1:
                    # spectrum.samples_mv[0] *= exp(- step * material.absorption_function_3d(start.x, start.y, start.z))
                
                emission_previous = emission
                emission.clear()
                length_traveled += step
                
        ####################################
        ############### MMM ################
        ####################################
        
        # you need smart sampling step to allow for scattering because of variable mfp
        elif material.use_step_function == 1 and material.use_scattering_function == 1:
            
            # sample point 
            emission_previous = material.emission_function(start, ray_direction, emission_previous,
                                                           world, ray, primitive,
                                                           world_to_primitive, primitive_to_world)
            
            # numerical integration:
            # stops when either collisions_max reached OR primitive boundary reached
            # calculate smart sampling step
            step_sampling = self._compute_step_sampling(material, start)
            
            # macroscopic cross section: sigma * density [m^{-1}]
            # reciprocal = mean free path between two collisions [m]
            sn = material.scattering_function_3d(start.x, start.y, start.z)   
            
            while collisions <= collisions_max:
            
                #######################
                ######### AAM #########
                #######################
                
                # sample the random free path from the pdf, knowing the TOTAL cross section sn
                #
                # CAUTION.
                # if uniform() == 0 or sn == 0 => fp = "infinite"
                try:
                    fp = - log(uniform()) / sn # [m]
                except:
                    fp = 1E+05 # [m]
                    
                # minimum between step_sampling and fp to properly resolve both emission and scattering-absirption
                # (HP. possibly scattering where step == 0, i.e. scattering > 0 although emission == 0)
                step = fmin(step_sampling, fp)
                
                start = new_point3d(
                    start.x + step * integration_direction.x,
                    start.y + step * integration_direction.y,
                    start.z + step * integration_direction.z
                )
                
                # check boundary has NOT been reached
                if primitive.contains(start) == True:
                  
                  emission = material.emission_function(start, ray_direction, emission_previous,
                                                        world, ray, primitive,
                                                        world_to_primitive, primitive_to_world)
                  
                  # trapezium rule integration
                  # HP: no wavelength dependence
                  spectrum.samples_mv[0] += 5E-01 * step * (emission.samples_mv[0] + emission_previous.samples_mv[0])
                  emission_previous = emission
                  emission.clear()
                    
                  # what happens next?
                  # if the fp < step_sampling:
                  # - compute emission at distance fp (although < step_sampling => over-sampling => conservative)
                  # - fp already traveled => compure scattering-vs-absorption (if active) and sample new direction
                  if fp < step_sampling:
                    
                    # macroscopic cross section: sigma * density [m^{-1}]
                    # reciprocal = mean free path between two collisions [m]
                    sn = material.scattering_function_3d(start.x, start.y, start.z)    
                    
                    #################################################
                    # 100% scattering, da modifcare se assorbimento #
                    #################################################
                    #if uniform() < scatt_vs_abs_probability: #forse qui ci vorrebbe la scattering_probability
                    
                    # isotropic scattering: how about Rayleigh scattering?
                    sphere_sampler = SphereSampler()
                    # sphere_sampler(1) return a list of 1 Vector3D and we take the 0th element
                    integration_direction = sphere_sampler(1)[0]
                    collisions += 1
                    
                    step_sampling = self._compute_step_sampling(material, start)
                    
                  else:
                    
                    # if step_sampling < fp then radiation sampling along the straight line
                    # until fp (i.e. collision point) is reached
                    #
                    # CAUTION. 
                    # step_sampling = step_sampling(x,y,z) varies along the line!
                    
                    length_traveled = step_sampling
                    
                    # update step_sampling in newly-computed point start                        
                    step_sampling = self._compute_step_sampling(material, start)
                    
                    # try defining stanalone variable residual_length = (fp-step_cumulative)
                    while fp - length_traveled > step_sampling:
                        
                        start = new_point3d(
                            start.x + step_sampling * integration_direction.x,
                            start.y + step_sampling * integration_direction.y,
                            start.z + step_sampling * integration_direction.z
                        )
                        
                        # check boundary has NOT been reached
                        if primitive.contains(start) == True:
                            
                          emission = material.emission_function(start, ray_direction, emission_previous,
                                                                world, ray, primitive,
                                                                world_to_primitive, primitive_to_world)
                          
                          # trapezium rule integration
                          # HP: no wavelength dependence
                          spectrum.samples_mv[0] += 5E-01 * step_sampling * (emission.samples_mv[0] + emission_previous.samples_mv[0])
                          emission_previous = emission
                          emission.clear()
                        
                        else:
                            # story ends
                            return spectrum
                        
                        length_traveled += step_sampling
                        
                        # update step_sampling in newly-computed point start                        
                        step_sampling = self._compute_step_sampling(material, start)
                        
                    # siamo arrivati a free path
                    step_sampling = fp - length_traveled
                    start = new_point3d(
                        start.x + step_sampling * integration_direction.x,
                        start.y + step_sampling * integration_direction.y,
                        start.z + step_sampling * integration_direction.z
                        )
                    
                    # check boundary has NOT been reached
                    if primitive.contains(start) == True:
                        
                      emission = material.emission_function(start, ray_direction, emission_previous,
                                                            world, ray, primitive,
                                                            world_to_primitive, primitive_to_world)
                      
                      # trapezium rule integration
                      # HP: no wavelength dependence
                      spectrum.samples_mv[0] += 5E-01 * step_sampling * (emission.samples_mv[0] + emission_previous.samples_mv[0])
                      emission_previous = emission
                      emission.clear()
                        
                    else:
                        # story ends
                        return spectrum
                    
                    step_sampling = self._compute_step_sampling(material, start)
                    
                    # macroscopic cross section: sigma * density [m^{-1}]
                    # reciprocal = mean free path between two collisions [m]
                    sn = material.scattering_function_3d(start.x, start.y, start.z)    
                    
                    # collision_probability = 1.0 - exp(- step * sn)
                    
                    # YES: collision condition is met
                    # (strict "<" sign - no "=" - because NO collision if collision_probability == 0)
                    
                    #################################################
                    # 100% scattering, da modifcare se assorbimento #
                    #################################################
                    #if uniform() < collision_probability: #forse qui ci vorrebbe la scattering_probability
                    
                    # isotropic scattering: how about Rayleigh scattering?
                    sphere_sampler = SphereSampler()
                    # sphere_sampler(1) return a list of 1 Vector3D and we take the 0th element
                    integration_direction = sphere_sampler(1)[0]
                    collisions += 1
                    
                else:
                  # story ends
                  return spectrum
                
        return spectrum
    
    cdef int _check_dimensions(self, Spectrum spectrum, int bins) except -1:
        if spectrum.samples.ndim != 1 or spectrum.samples.shape[0] != bins:
            raise ValueError("Spectrum returned by emission function has the wrong number of samples.")
            
    cdef double _compute_step_sampling(self, InhomogeneousVolumeEmitter material, Point3D point) except -1:
        cdef: double step_sampling = material.step_function_3d(point.x, point.y, point.z)
        if step_sampling == 0.0: # <=> emission == 0.0 => maximum step_sampling
            return material.step_max
        return step_sampling
            
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
