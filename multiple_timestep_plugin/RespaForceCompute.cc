// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: NathanDavisBarrett

/*! \file RespaForceCompute.cc
    \brief Defines the RespaForceCompute class
*/



#include "RespaForceCompute.h"

#ifdef ENABLE_MPI
#include <hoomd/Communicator.h>
#endif

#include <iostream>
using namespace std;

namespace py = pybind11;

#include <memory>

/*! \param sysdef System to compute forces on
    \post The Compute is initialized and all memory needed for the forces is allocated
    \post \c force and \c virial GPUarrays are initialized
    \post All forces are initialized to 0
*/
RespaForceCompute::RespaForceCompute(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group)
        : ForceCompute(sysdef)
{
    m_group = group;
}

/*! Frees allocated memory
*/
RespaForceCompute::~RespaForceCompute()
{

}

//Keep this around for now so we can use it as a pattern for compute forces.
// /*! Sums the force of a particle group calculated by the last call to compute() and returns it.
//*/
//
//vec3<double> ForceCompute::calcForceGroup(std::shared_ptr<ParticleGroup> group)
//{
//    unsigned int group_size = group->getNumMembers();
//    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::read);
//
//    vec3<double> f_total = vec3<double>();
//
//    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
//    {
//        unsigned int j = group->getMemberIndex(group_idx);
//
//        f_total += (vec3<double>)h_force.data[j];
//    }
//#ifdef ENABLE_MPI
//    if (m_comm)
//        {
//        // reduce potential energy on all processors
//        MPI_Allreduce(MPI_IN_PLACE, &f_total, 3, MPI_DOUBLE, MPI_SUM, m_exec_conf->getMPICommunicator());
//        }
//#endif
//    return vec3<double>(f_total);
//}

void export_RespaForceCompute(py::module& m)
{
    py::class_< RespaForceCompute, std::shared_ptr<RespaForceCompute> >(m,"RespaForceCompute",py::base<FroceCompute>())
            .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >())
            .def("getForce", &ForceCompute::getForce)
            .def("getTorque", &ForceCompute::getTorque)
            .def("getVirial", &ForceCompute::getVirial)
            .def("getEnergy", &ForceCompute::getEnergy)
            .def("calcEnergyGroup", &ForceCompute::calcEnergyGroup)
            .def("calcForceGroup", &ForceCompute::calcForceGroup)
            .def("calcVirialGroup", &ForceCompute::calcVirialGroup)
            ;
}
