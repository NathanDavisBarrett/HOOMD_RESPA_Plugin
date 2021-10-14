// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: NathanDavisBarrett
#include <hoomd/ForceCompute.h>
#include <hoomd/Index1D.h>
#include <hoomd/ParticleGroup.h>

#include <hoomd/GlobalArray.h>
#include <hoomd/GlobalArray.h>

#ifdef ENABLE_CUDA
#include <hoomd/ParticleData.cuh>
#endif

#ifdef ENABLE_MPI
#include <hoomd/Communicator.h>
#endif

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

/*! \file RespaForceCompute.h
    \brief Declares the RespaForceCompute class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __RESPAFORCECOMPUTE_H__
#define __RESPAFORCECOMPUTE_H__

//! Handy structure for passing the force arrays around in the Respa Algorithm
/*! \c fx, \c fy, \c fz have length equal to the number of particles in the specified group and store the x,y,z
    components of the force on that particle. \a pe is also included as the potential energy
    for each particle in the specified group, if it can be defined for the force. \a virial is the per particle virial.

    The per particle potential energy is defined such that \f$ \sum_i^N \mathrm{pe}_i = V_{\mathrm{total}} \f$

    The per particle virial is a upper triangular 3x3 matrix that is defined such
    that
    \f$ \sum_k^N \left(\mathrm{virial}_{ij}\right)_k = \sum_k^N \sum_{l>k} \frac{1}{2} \left( \vec{f}_{kl,i} \vec{r}_{kl,j} \right) \f$

    \ingroup data_structs
*/

class PYBIND11_EXPORT RespaForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        RespaForceCompute(std::shared_ptr<SystemDefinition>
        sysdef,
        std::shared_ptr <ParticleGroup> group
        );

        //! Destructor
        virtual ~RespaForceCompute();

    protected:
        std::shared_ptr<ParticleGroup> m_group;

        //! Actually perform the computation of the forces
        /*! This is pure virtual here. Sub-classes must implement this function. It will be called by
            the base class compute() when the forces need to be computed.
            \param timestep Current time step
        */
        virtual void computeForces(unsigned int timestep) {}

    };

//! Exports the ForceCompute class to python
#ifndef NVCC
void export_RespaForceCompute(pybind11::module& m);
#endif

#endif
