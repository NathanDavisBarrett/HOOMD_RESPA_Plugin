//
// Created by nathan on 8/12/21.
//

#ifndef RESPAPLUGIN_RespaINTEGRATOR_CUH
#define RESPAPLUGIN_RespaINTEGRATOR_CUH

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

/*! \file RespaIntegrator.cuh
    \brief Declaration of CUDA kernels for RespaIntegrator
*/

// A C API call to run a CUDA kernel is needed for RespaIntegratorGPU to call
//! Zeros velocities on the GPU
extern "C" hipError_t gpu_zero_velocities(Scalar4* d_vel, unsigned int N);

#endif //RESPAPLUGIN_RespaINTEGRATOR_CUH
