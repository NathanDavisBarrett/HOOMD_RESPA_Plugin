// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: NathanDavisBarrett

#ifndef __RESPA_POTENTIAL_PAIR_H__
#define __RESPA_POTENTIAL_PAIR_H__


#include <iostream>
#include <stdexcept>
#include <memory>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/numpy.h>

#include "RespaForceCompute.h"

#include <hoomd/HOOMDMath.h>
#include <hoomd/Index1D.h>
#include <hoomd/GlobalArray.h>
#include <hoomd/ForceCompute.h>
#include <hoomd/md/PotentialPair.h>
#include <hoomd/md/NeighborList.h>
#include <hoomd/GSDShapeSpecWriter.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_MPI
#include <hoomd/Communicator.h>
#endif


/*! \file RespaPotentialPair.h
    \brief Defines the template class for standard RESPA pair potentials
    \details The heart of the code that computes pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing RESPA pair potentials
/*! <b>Overview:</b>
    RespaPotentialPair computes standard pair potentials (and forces) between all particle pairs in a specified group. It
    employs the use of a neighbor list to limit the number of computations done to only those particles with the
    cutoff radius of each other. The computation of the actual V(r) is not performed directly by this class, but
    by an evaluator class (e.g. EvaluatorPairLJ) which is passed in as a template parameter so the computations
    are performed as efficiently as possible.

    RespaPotentialPair handles most of the gory internal details common to all standard pair potentials.
     - A cutoff radius to be specified per particle type pair
     - The energy can be globally shifted to 0 at the cutoff
     - XPLOR switching can be enabled
     - Per type pair parameters are stored and a set method is provided
     - Logging methods are provided for the energy
     - And all the details about looping through the particles, computing dr, computing the virial, etc. are handled

    A note on the design of XPLOR switching:
    We need to be able to handle smooth XPLOR switching in systems of mixed LJ/WCA particles. There are three modes to
    enable all of the various use-cases:
     - Mode 1: No shifting. All pair potentials are computed as is and not shifted to 0 at the cutoff.
     - Mode 2: Shift everything. All pair potentials (no matter what type pair) are shifted so they are 0 at the cutoff
     - Mode 3: XPLOR switching enabled. A r_on value is specified per type pair. When r_on is less than r_cut, normal
       XPLOR switching will be applied to the unshifted potential. When r_on is greater than r_cut, the energy will
       be shifted. In this manner, a valid r_on value can be given for the LJ interactions and r_on > r_cut can be set
       for WCA (which will then be shifted).

    XPLOR switching gets significantly more complicated for all pair potentials when shifted potentials are used. Thus,
    the combination of XPLOR switching + shifted potentials will not be supported to avoid slowing down the calculation
    for everyone.

    <b>Implementation details</b>

    rcutsq, ronsq, and the params are stored per particle type pair. It wastes a little bit of space, but benchmarks
    show that storing the symmetric type pairs and indexing with Index2D is faster than not storing redundant pairs
    and indexing with Index2DUpperTriangular. All of these values are stored in GlobalArray
    for easy access on the GPU by a derived class. The type of the parameters is defined by \a param_type in the
    potential evaluator class passed in. See the appropriate documentation for the evaluator for the definition of each
    element of the parameters.

    For profiling and logging, PotentialPair needs to know the name of the potential. For now, that will be queried from
    the evaluator. Perhaps in the future we could allow users to change that so multiple pair potentials could be logged
    independently.

    \sa export_PotentialPair()
*/

template < class evaluator >
class RespaPotentialPair: public PotentialPair<evaluator>, public RespaForceCompute
{
public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Construct the pair potential
    RespaPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<NeighborList> nlist,
                  std::shared_ptr <ParticleGroup> group,
                  const std::string& log_suffix="");
    //! Destructor
    virtual ~RespaPotentialPair();

    //Due to the multiple inheritance paths, these lines are needed to clarify any ambiguity.
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf =  this->PotentialPair<evaluator>::m_exec_conf;
    std::shared_ptr<Profiler> m_prof = this->PotentialPair<evaluator>::m_prof;
    const std::shared_ptr<ParticleData> m_pdata = this->PotentialPair<evaluator>::m_pdata;
    //GlobalArray<Scalar4> m_force = this->PotentialPair<evaluator>::m_force;                           //TODO: For some reason the calculated forces are being seen as zero. I assume it's because I'm referencing a different m_forces between the compute call and the reference call.
    GlobalArray<Scalar4> m_force = this->RespaForceCompute::m_force;
    GlobalArray<Scalar>  m_virial = this->PotentialPair<evaluator>::m_virial;
    unsigned int m_virial_pitch = this->PotentialPair<evaluator>::m_virial_pitch;

protected:

    //! Actually compute the forces
    virtual void computeForces(unsigned int timestep);

};

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param log_suffix Name given to this instance of the force
*/
template < class evaluator >
RespaPotentialPair< evaluator >::RespaPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                                          std::shared_ptr<NeighborList> nlist,
                                          std::shared_ptr <ParticleGroup> group,
                                          const std::string& log_suffix)
        : PotentialPair<evaluator>(sysdef,nlist,log_suffix), RespaForceCompute(sysdef, group)
{
    m_exec_conf->msg->notice(5) << "Constructing RespaPotentialPair<" << evaluator::getName() << ">" << std::endl;
    m_exec_conf->msg->warning() << "RespaPotentialPair m_force is located at:" << &m_force << "\n";
}

template< class evaluator >
RespaPotentialPair< evaluator >::~RespaPotentialPair()
{
    m_exec_conf->msg->notice(5) << "Destroying RespaPotentialPair<" << evaluator::getName() << ">" << std::endl;
}

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is called to ensure
    that it is up to date before proceeding.

    RESPA Edit: This is the same function as defined in PotentialPair.h, the only difference is that here, only the
    particles in the specified group will be itterated over.

    \param timestep specifies the current time step of the simulation
*/
template< class evaluator >
void RespaPotentialPair< evaluator >::computeForces(unsigned int timestep)
{
    m_exec_conf->msg->warning() << "@@@@@@@@@@@ compute call received @@@@@@@@@@@";
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(this->m_prof_name);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(this->m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(this->m_nlist->getNListArray(), access_location::host, access_mode::read);
//     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<unsigned int> h_head_list(this->m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    //force arrays
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar>  h_virial(m_virial,access_location::host, access_mode::overwrite);


    const BoxDim& box = m_pdata->getGlobalBox();
    ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(this->m_params, access_location::host, access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // for each particle
    for (int groupi = 0; groupi < (int)m_group->getNumMembersGlobal(); groupi++)
    {
        // Extract the actual particle index from the group index and assign it to "i"
        int tagi = m_group->getMemberTag(groupi);
        int i = m_pdata->getRTag(tagi);

        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter and charge (if needed)
        Scalar di = Scalar(0.0);
        Scalar qi = Scalar(0.0);
        if (evaluator::needsDiameter())
            di = h_diameter.data[i];
        if (evaluator::needsCharge())
            qi = h_charge.data[i];

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++) {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // If the nighboring particle is not in the group, continue.
            if (!(m_group->isMember(j))) {
                continue;
            }

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access diameter and charge (if needed)
            Scalar dj = Scalar(0.0);
            Scalar qj = Scalar(0.0);
            if (evaluator::needsDiameter())
                dj = h_diameter.data[j];
            if (evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // get parameters for this type pair
            unsigned int typpair_idx = this->m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];
            Scalar ronsq = Scalar(0.0);
            if (this->m_shift_mode == this->xplor)
                ronsq = h_ronsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (this->m_shift_mode == this->shift)
                energy_shift = true;
            else if (this->m_shift_mode == this->xplor) {
                if (ronsq > rcutsq)
                    energy_shift = true;
            }

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, dj);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

            if (evaluated) {
                // modify the potential for xplor shifting
                if (this->m_shift_mode == this->xplor) {
                    if (rsq >= ronsq && rsq < rcutsq) {
                        // Implement XPLOR smoothing (FLOPS: 16)
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv =
                                Scalar(1.0) / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq *
                                   (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq) * xplor_denom_inv;
                        Scalar ds_dr_divr = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a +
                        // But this is verified correct via plotting
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                    }
                }

                Scalar force_div2r = force_divr * Scalar(0.5);
                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx * force_divr;
                pei += pair_eng * Scalar(0.5);
                if (compute_virial) {
                    virialxxi += force_div2r * dx.x * dx.x;
                    virialxyi += force_div2r * dx.x * dx.y;
                    virialxzi += force_div2r * dx.x * dx.z;
                    virialyyi += force_div2r * dx.y * dx.y;
                    virialyzi += force_div2r * dx.y * dx.z;
                    virialzzi += force_div2r * dx.z * dx.z;
                }

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                // only add force to local particles
                if (third_law && j < m_pdata->getN()) {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x * force_divr;
                    h_force.data[mem_idx].y -= dx.y * force_divr;
                    h_force.data[mem_idx].z -= dx.z * force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                    if (compute_virial) {
                        h_virial.data[0 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.x;
                        h_virial.data[1 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.y;
                        h_virial.data[2 * m_virial_pitch + mem_idx] += force_div2r * dx.x * dx.z;
                        h_virial.data[3 * m_virial_pitch + mem_idx] += force_div2r * dx.y * dx.y;
                        h_virial.data[4 * m_virial_pitch + mem_idx] += force_div2r * dx.y * dx.z;
                        h_virial.data[5 * m_virial_pitch + mem_idx] += force_div2r * dx.z * dx.z;
                    }
                }
            }
        }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        if (i % 100 == 0) {
            m_exec_conf->msg->warning() << "!PotPr.h>> These should not be zero:" << fi.x << " " << fi.y << " " << fi.z
                                        << "\n";
        }

        if (compute_virial)
        {
            h_virial.data[0*m_virial_pitch+mem_idx] += virialxxi;
            h_virial.data[1*m_virial_pitch+mem_idx] += virialxyi;
            h_virial.data[2*m_virial_pitch+mem_idx] += virialxzi;
            h_virial.data[3*m_virial_pitch+mem_idx] += virialyyi;
            h_virial.data[4*m_virial_pitch+mem_idx] += virialyzi;
            h_virial.data[5*m_virial_pitch+mem_idx] += virialzzi;
        }
    }

    if (m_prof) m_prof->pop();
}

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template < class T > void export_RespaPotentialPair(pybind11::module& m, const std::string& name)
{
    pybind11::print("ADDED RESPAFORCECOMPUTE BASE REFERENCE.");
    pybind11::class_<T, std::shared_ptr<T> > respapotentialpair(m, name.c_str(), pybind11::base<RespaForceCompute>());
    respapotentialpair.def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, std::shared_ptr <ParticleGroup>, const std::string& >())
            .def("setParams", &T::setParams)
            .def("setRcut", &T::setRcut)
            .def("setRon", &T::setRon)
            .def("setShiftMode", &T::setShiftMode)
            .def("computeEnergyBetweenSets", &T::computeEnergyBetweenSetsPythonList)
            .def("slotWriteGSDShapeSpec", &T::slotWriteGSDShapeSpec)
            .def("connectGSDShapeSpec", &T::connectGSDShapeSpec)
            ;
}
#endif // __RESPA_POTENTIAL_PAIR_H__
