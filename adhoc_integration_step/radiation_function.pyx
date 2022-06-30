# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

from raysect.optical cimport Point3D, Vector3D, Spectrum, World, Ray, Primitive, AffineMatrix3D
from raysect.optical.material.emitter cimport InhomogeneousVolumeEmitter, NumericalIntegrator
from cherab.core.math.function cimport Function3D, autowrap_function3d
from libc.math cimport M_PI
import cython


cdef class RadiationFunction(InhomogeneousVolumeEmitter):
    """
    A general purpose radiation material.

    Radiates power over 4 pi according to the supplied 3D radiation
    function. Note that this model ignores the spectral range of the
    observer. The power specified will be spread of the entire
    observable spectral range. Useful for calculating total radiated
    power loads on reactor wall components.

    Note that the function will be evaluated in the local space of the
    primitive to which this material is attached. For radiation
    functions defined in a different coordinate system, consider
    wrapping this in a VolumeTransform material to ensure the function
    evaluation takes place in the correct coordinate system.

    Further parameters added to increase functionalities (e.g. non-uniform sampling).

    CAUTION.
    
    - Scattering NOT YET IMPLEMENTED
    - When implemented, would be QUALITATIVE only: reverse ray-tracing
      will somewhat "bias" trajectories (i.e. free path computed with
      cross-section at the "future" point in space...)

    :param Function3D radiation_function: A 3D radiation function that specifies the amount of radiation
      to be radiated at a given point, :math:`\phi(x, y, z)` [W/m^3].
    :param float step: The scale length for integration of the radiation function.
    :param int use_step_function: If True (or 1) activates non-uniform sampling.
    :param Function3D step_function_3d: A 3D function that specifies the non-uniform sampling step
      at a given point, :math:`\Delta s(x,y,z)` [m^{-1}].
    :param int step_max: Maximum sampling step allowed in non-uniform sampling, :math:`\Delta s_{max}` [m].
    :param int use_absorption_function: If True (or 1) activates self-absorption.
    :param Function3D absorption_function_3d: A 3D function that specifies the macroscopic absorption
      cross section at a given point, :math:`\Sigma_{abs}(x,y,z)` [m].
    :param int use_scattering_function: If True (or 1) activates self-scattering.
    :param Function3D scattering_function_3d: A 3D function that specifies the macroscopic scattering
      cross section at a given point, :math:`\Sigma_{sct}(x,y,z)` [m].
    :param int collisions_max: Maximum number of self-scattering events (collisions).

    .. code-block:: pycon

       >>> import numpy as np
       >>> from cherab.tools.emitters import RadiationFunction
       >>>
       >>> # define your own 3D radiation function and insert it into this class
       >>> def rad_function_3d(x, y, z):
               r = np.sqrt(x**2 + y**2 + z**2)
               if r == 0: return 0
               else:      return np.min([1E+00, r])
       >>>
       >>> # define your own 3D step function and insert it into this class
       >>> def step_function_3d(x, y, z):
               r = np.sqrt(x**2 + y**2 + z**2)
               if r == 0: return 1E-05
               else:      return 1E-02 * np.min([1E+00, r])
       >>>
       >>> # define your own 3D absorption function and insert it into this class
       >>> def abs_function_3d(x, y, z): return 1E-01 * rad_function_3d(x,y,z)
       >>>
       >>> radiation_emitter = RadiationFunction(radiation_function      = rad_function_3d,
                                                 step                    = np.inf,
                                                 use_step_function       = True,
                                                 step_function_3d        = step_function_3d,
                                                 step_max                = 1E-02,
                                                 use_absorption_function = True,
                                                 absorption_function_3d  = abs_function_3d,
                                                 use_scattering_function = False,
                                                 scattering_function_3d  = None,
                                                 collisions_max          = 0
                                                )
    """

    cdef:
        readonly Function3D radiation_function
        readonly Function3D absorption_function_3d   # MMM
        readonly Function3D scattering_function_3d   # MMM
        readonly Function3D step_function_3d         # MMM
        readonly int use_absorption_function         # MMM
        readonly int use_scattering_function         # MMM
        readonly int use_step_function               # MMM
        readonly int collisions_max                  # MMM
        readonly float step_max                      # MMM

    def __init__(self, radiation_function,                              # emission
                       use_step_function,       step_function_3d,       # non-uniform sampling
                       use_absorption_function, absorption_function_3d, # absorption
                       use_scattering_function, scattering_function_3d, # scattering
                       collisions_max = 100,    step_max = 0.1,  step = 0.1):

        super().__init__(NumericalIntegrator(step = step))
        # radiation emission
        self.radiation_function      = autowrap_function3d(radiation_function)
        # non-uniform sampling
        self.use_step_function       = use_step_function
        self.step_function_3d        = autowrap_function3d(step_function_3d)
        self.step_max                = step_max
        # absorption
        self.use_absorption_function = use_absorption_function
        self.absorption_function_3d  = autowrap_function3d(absorption_function_3d)
        # scattering
        self.use_scattering_function = use_scattering_function
        self.scattering_function_3d  = autowrap_function3d(scattering_function_3d)
        self.collisions_max          = collisions_max

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum emission_function(self, Point3D point, Vector3D direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix3D world_to_local, AffineMatrix3D local_to_world):

        cdef int index
        cdef double wvl_range = ray.max_wavelength - ray.min_wavelength
        cdef double emission

        emission = self.radiation_function.evaluate(point.x, point.y, point.z) / (4 * M_PI * wvl_range)

        for index in range(spectrum.bins):
            spectrum.samples_mv[index] += emission
        return spectrum
