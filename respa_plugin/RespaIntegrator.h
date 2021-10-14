//
// Created by nathan on 8/12/21.
//

#ifndef RESPAPLUGIN_RESPAINTEGRATOR_H
#define RESPAPLUGIN_RESPAINTEGRATOR_H

/*! \file RespaIntegrator.h
    \brief Declaration of RespaIntegrator
*/

#include <hoomd/Integrator.h>
#include <vector>
#include <utility> //std::pair
#include <string>

#include "RespaForceCompute.h"

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
class RespaIntegrator : public Integrator
        {
        private:
            std::vector<std::pair<std::shared_ptr<RespaForceCompute>, int>> m_respa_forces;

            const int VEL_STEP = 0;
            const int POS_STEP = 1;

            std::vector<int> m_respa_step_types;
            std::vector<std::shared_ptr<RespaForceCompute>> m_respa_step_force_computes;
            std::vector<Scalar> m_respa_step_force_scaling_factors;
            std::vector<Scalar> m_respa_step_vel_scaling_factors;


        protected:
            bool m_prepared;              //!< True if preprun had been called

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
            RespaIntegrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);

            //! Destructor
            ~RespaIntegrator();

            //! Take one timestep forward
            void update(unsigned int timestep);

            /// Sets the profiler for the compute to use
            void setProfiler(std::shared_ptr<Profiler> prof);

            /// Change the timestep
            void setDeltaT(Scalar deltaT);

            /// Get the number of degrees of freedom granted to a given group
            unsigned int getNDOF(std::shared_ptr<ParticleGroup> group);

            /// Get the number of degrees of freedom granted to a given group
            unsigned int getRotationalNDOF(std::shared_ptr<ParticleGroup> group);

            /// Set the anisotropic mode of the integrator
            void setAnisotropicMode(AnisotropicMode mode);

            /// get the anisotropic mode of the integrator
            bool getAnisotropicMode();

            Scalar calculateForceScalingFactor(int numSubsteps);

            Scalar calculateVelScalingFactor(int numSubsteps);

            void addSubstep(int stepType, std::shared_ptr<RespaForceCompute> forceCompute, int numSubsteps);

            void createSubsteps(std::vector<std::pair<std::shared_ptr<RespaForceCompute>, int>>, int);

            /// Prepare for the run
            void prepRun(unsigned int timestep);

            /// helper function to compute net force/virial
            void computeNetForce(unsigned int timestep);

            /// Add a new force/frequency pair.
            void addForce(std::shared_ptr<RespaForceCompute>, int);

        protected:
            AnisotropicMode m_aniso_mode; //!< Anisotropic mode for this integrator
        };

//! Export the RespaIntegrator class to python
void export_RespaIntegrator(pybind11::module& m);

// Third, this class offers a GPU accelerated method in order to demonstrate how to include CUDA
// code in pluins we need to declare a separate class for that (but only if ENABLE_HIP is set)

#ifdef ENABLE_HIP

//! A GPU accelerated nonsense particle Integrator written to demonstrate how to write a plugin w/ CUDA
//! code
/*! This Integrator simply sets all of the particle's velocities to 0 (on the GPU) when update() is
 * called.
 */
class RespaIntegratorGPU : public RespaIntegrator
        {
        public:
            //! Constructor
            RespaIntegratorGPU(std::shared_ptr<SystemDefinition> sysdef);

            //! Take one timestep forward
            virtual void update(uint64_t timestep);


        };

//! Export the ExampleIntegratorGPU class to python
void export_RespaIntegratorGPU(pybind11::module& m);

#endif // ENABLE_HIP

#endif //RespaPLUGIN_RespaINTEGRATOR_H
