//
// Created by nathan on 8/12/21.
//

#ifndef MULTIPLETIMESTEPPLUGIN_MULTIPLETIMESTEPINTEGRATOR_H
#define MULTIPLETIMESTEPPLUGIN_MULTIPLETIMESTEPINTEGRATOR_H

/*! \file MultipleTimestepIntegrator.h
    \brief Declaration of MultipleTimestepIntegrator
*/

#include <hoomd/Integrator.h>
#include <vector>
#include <utility> //std::pair
#include "RespaStep.h"
#include "RespaVelStep.h"
#include "RespaPosStep.h"

// pybind11 is used to create the python bindings to the C++ object,
// but not if we are compiling GPU kernels
#ifndef __HIPCC__
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND
// ONLY IF hoomd_config.h is included first) For example: #include <hoomd/Integrator.h>

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a
// template here, there are no restrictions on what a template can do

//! A nonsense particle Integrator written to demonstrate how to write a plugin
/*! This Integrator simply sets all of the particle's velocities to 0 when update() is called.
 */
class MultipleTimestepIntegrator : public Integrator
        {
        private:
            std::vector<std::pair<std::shared_ptr<ForceCompute>, int>> m_respa_forces;
            std::vector<*RespaStep> m_respa_steps;

        public:
            /** Anisotropic integration mode: Automatic (detect whether
                aniso forces are defined), Anisotropic (integrate
                rotational degrees of freedom regardless of whether
                anything is defining them), and Isotropic (don't integrate
                rotational degrees of freedom)
            */
            enum AnisotropicMode
                    {
                Automatic,
                Anisotropic,
                Isotropic
                    };

            //! Constructor
            MultipleTimestepIntegrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);

            //! Destructor
            ~MultipleTimestepIntegrator();

            //! Take one timestep forward
            void update(uint64_t timestep);

            /// Sets the profiler for the compute to use
            void setProfiler(std::shared_ptr<Profiler> prof);

            /// Change the timestep
            void setDeltaT(Scalar deltaT);

            /// Get the number of degrees of freedom granted to a given group
            Scalar getTranslationalDOF(std::shared_ptr<ParticleGroup> group);

            /// Get the number of degrees of freedom granted to a given group
            Scalar getRotationalDOF(std::shared_ptr<ParticleGroup> group);

            /// Set the anisotropic mode of the integrator
            void setAnisotropicMode(const std::string& mode);

            /// Set the anisotropic mode of the integrator
            const std::string getAnisotropicMode();

            void createSubsteps(vector<std::pair<std::shared_ptr<ForceCompute>, int>>, int);

            /// Prepare for the run
            void prepRun(uint64_t timestep);

            /// helper function to compute net force/virial
            void computeNetForce(uint64_t timestep);

            /// Add a new force/frequency pair.
            void addForce(std::pair<std::shared_ptr<ForceCompute>, int>);
        };

//! Export the MultipleTimestepIntegrator class to python
void export_MultipleTimestepIntegrator(pybind11::module& m);

// Third, this class offers a GPU accelerated method in order to demonstrate how to include CUDA
// code in pluins we need to declare a separate class for that (but only if ENABLE_HIP is set)

#ifdef ENABLE_HIP

//! A GPU accelerated nonsense particle Integrator written to demonstrate how to write a plugin w/ CUDA
//! code
/*! This Integrator simply sets all of the particle's velocities to 0 (on the GPU) when update() is
 * called.
 */
class MultipleTimestepIntegratorGPU : public MultipleTimestepIntegrator
        {
        public:
            //! Constructor
            MultipleTimestepIntegratorGPU(std::shared_ptr<SystemDefinition> sysdef);

            //! Take one timestep forward
            virtual void update(uint64_t timestep);


        };

//! Export the ExampleIntegratorGPU class to python
void export_MultipleTimestepIntegratorGPU(pybind11::module& m);

#endif // ENABLE_HIP

#endif //MULTIPLETIMESTEPPLUGIN_MULTIPLETIMESTEPINTEGRATOR_H
