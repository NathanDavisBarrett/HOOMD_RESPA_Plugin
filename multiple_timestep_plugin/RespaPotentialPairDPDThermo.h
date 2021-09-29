//
// Created by nathan on 9/25/21.
//

#ifndef MULTIPLE_TIMESTEP_PLUGIN_RESPAPOTENTIALPAIRDPDTHERMO_H
#define MULTIPLE_TIMESTEP_PLUGIN_RESPAPOTENTIALPAIRDPDTHERMO_H
/*
#include <hoomd/md/PotentialPair.h>
#include <hoomd/md/PotentialPairDPDThermo.h>
#include <hoomd/md/EvaluatorPairDPDThermo.h>
#include <hoomd/md/EvaluatorPairDPDLJThermo.h>
#include <hoomd/md/NeighborList.h>
#include <hoomd/Variant.h>

#include "RespaPotentialPair.h"

/*! \file RespaPotentialPairDPDThermo.h
    \brief Etends the PotentialPairDPDThermo class to operate with the RespaForceCompute methodology

*./

template < class evaluator >
class RespaPotentialPairDPDThermo : public PotentialPairDPDThermo<evaluator>, public RespaPotentialPair<evaluator>
{
public:
    typedef typename evaluator::param_type param_type;

    RespaPotentialPairDPDThermo(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<NeighborList> nlist,
                    std::shared_ptr <ParticleGroup> group,
                    const std::string& log_suffix=""): PotentialPairDPDThermo<evaluator>(sysdef,nlist,log_suffix),
                            RespaPotentialPair<evaluator>(sysdef,nlist,group,log_suffix){};

    virtual ~RespaPotentialPairDPDThermo() { };

    //Due to the multiple inheritance paths, these lines are needed to clarify any ambiguity.
    std::shared_ptr<NeighborList> m_nlist = this->PotentialPairDPDThermo<evaluator>::m_nlist;
    std::shared_ptr<Profiler> m_prof = this->PotentialPairDPDThermo<evaluator>::m_prof;
    std::string m_prof_name = this->PotentialPairDPDThermo<evaluator>::m_prof_name;
    const std::shared_ptr<ParticleData> m_pdata = this->PotentialPairDPDThermo<evaluator>::m_pdata;
    GlobalArray<Scalar4> m_force = this->PotentialPairDPDThermo<evaluator>::m_force;
    GlobalArray<Scalar>  m_virial = this->PotentialPairDPDThermo<evaluator>::m_virial;
    GlobalArray<Scalar> m_rcutsq = this->PotentialPairDPDThermo<evaluator>::m_rcutsq;
    GlobalArray<Scalar> m_ronsq = this->PotentialPairDPDThermo<evaluator>::m_ronsq;
    GlobalArray<param_type> m_params = this->PotentialPairDPDThermo<evaluator>::m_params;
    Index2D m_typpair_idx = this->PotentialPairDPDThermo<evaluator>::m_typpair_idx;
    Scalar m_deltaT = this->PotentialPairDPDThermo<evaluator>::m_deltaT;
    unsigned int m_virial_pitch = this->PotentialPairDPDThermo<evaluator>::m_virial_pitch;

protected:
    virtual void computeForces(unsigned int timestep);

};

template< class evaluator >
void RespaPotentialPairDPDThermo<evaluator>::computeForces(unsigned int timestep) {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // start the profile for this compute
    if (this->m_prof) this->m_prof->push(this->m_prof_name);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(this->m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(this->m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(this->m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(this->m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);

    //force arrays
    ArrayHandle<Scalar4> h_force(this->m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar>  h_virial(this->m_virial,access_location::host, access_mode::overwrite);

    const BoxDim& box = this->m_pdata->getBox();
    ArrayHandle<Scalar> h_ronsq(this->m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(this->m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(this->m_params, access_location::host, access_mode::read);

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data,0,sizeof(Scalar4)*this->m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*this->m_virial.getNumElements());

    // for each particle
    for (int groupi = 0; groupi < (int)this->m_group->getNumMembersGlobal(); groupi++)
    {
        // Extract the actual particle index from the group index and assign it to "i"
        int tagi = this->m_group->getMemberTag(groupi);
        int i = m_pdata->getRTag(tagi);

        // access the particle's position, velocity, and type (MEM TRANSFER: 7 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 vi = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);

        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const unsigned int head_i = h_head_list.data[i];

        // sanity check
        assert(typei < this->m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0,0,0);
        Scalar pei = 0.0;
        Scalar viriali[6];
        for (unsigned int l = 0; l < 6; l++)
            viriali[l] = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
        {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[head_i + k];
            assert(j < this->m_pdata->getN() + this->m_pdata->getNGhosts() );

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // calculate dv_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 vj = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
            Scalar3 dv = vi - vj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < this->m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            //calculate the drag term r \dot v
            Scalar rdotv = dot(dx, dv);

            // get parameters for this type pair
            unsigned int typpair_idx = this->m_typpair_idx(typei, typej);
            param_type param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            bool energy_shift = false;
            if (this->PotentialPairDPDThermo<evaluator>::m_shift_mode == this->shift)
                energy_shift = true;

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar force_divr_cons = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);

            // Special Potential Pair DPD Requirements
            const Scalar currentTemp = this->m_T->getValue(timestep);

            // set seed using global tags
            unsigned int tagi = h_tag.data[i];
            unsigned int tagj = h_tag.data[j];
            eval.set_seed_ij_timestep(this->m_seed,tagi,tagj,timestep);
            eval.setDeltaT(this->m_deltaT);
            eval.setRDotV(rdotv);
            eval.setT(currentTemp);

            bool evaluated = eval.evalForceEnergyThermo(force_divr, force_divr_cons, pair_eng, energy_shift);

            if (evaluated)
            {
                // compute the virial (FLOPS: 2)
                Scalar pair_virial[6];
                pair_virial[0] = Scalar(0.5) * dx.x * dx.x * force_divr_cons;
                pair_virial[1] = Scalar(0.5) * dx.x * dx.y * force_divr_cons;
                pair_virial[2] = Scalar(0.5) * dx.x * dx.z * force_divr_cons;
                pair_virial[3] = Scalar(0.5) * dx.y * dx.y * force_divr_cons;
                pair_virial[4] = Scalar(0.5) * dx.y * dx.z * force_divr_cons;
                pair_virial[5] = Scalar(0.5) * dx.z * dx.z * force_divr_cons;


                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx*force_divr;
                pei += pair_eng * Scalar(0.5);
                for (unsigned int l = 0; l < 6; l++)
                    viriali[l] += pair_virial[l];

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law)
                {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x*force_divr;
                    h_force.data[mem_idx].y -= dx.y*force_divr;
                    h_force.data[mem_idx].z -= dx.z*force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                    for (unsigned int l = 0; l < 6; l++)
                        h_virial.data[l * this->m_virial_pitch + mem_idx] += pair_virial[l];
                }
            }
        }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        for (unsigned int l = 0; l < 6; l++)
            h_virial.data[l * this->m_virial_pitch + mem_idx] += viriali[l];
    }

    if (this->m_prof) this->m_prof->pop();
}

template < class T, class Base > void export_RespaPotentialPairDPDThermo(pybind11::module& m, const std::string& name)
{

    pybind11::class_<T, std::shared_ptr<T> >(m, name.c_str(), pybind11::base< Base >())
            .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, std::shared_ptr <ParticleGroup>, const std::string& >())
            .def("setSeed", &T::setSeed)
            .def("setT", &T::setT)
            ;
}
*/
#endif //MULTIPLE_TIMESTEP_PLUGIN_RESPAPOTENTIALPAIRDPDTHERMO_H
