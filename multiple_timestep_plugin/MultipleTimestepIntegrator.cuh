//
// Created by nathan on 8/12/21.
//

#ifndef MULTIPLETIMESTEPPLUGIN_MULTIPLETIMESTEPINTEGRATOR_CUH
#define MULTIPLETIMESTEPPLUGIN_MULTIPLETIMESTEPINTEGRATOR_CUH

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

/*! \file MultipleTimestepIntegrator.cuh
    \brief Declaration of CUDA kernels for MultipleTimestepIntegrator
*/

// A C API call to run a CUDA kernel is needed for MultipleTimestepIntegratorGPU to call
//! Zeros velocities on the GPU
extern "C" hipError_t gpu_zero_velocities(Scalar4* d_vel, unsigned int N);

#endif //MULTIPLETIMESTEPPLUGIN_MULTIPLETIMESTEPINTEGRATOR_CUH
