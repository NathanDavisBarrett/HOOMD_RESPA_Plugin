//
// Created by nathan on 8/12/21.
//

#include "RespaVelStep.h"

/ Constructor
RespaVelStep::RespaVelStep(std::shared_ptr<ForceCompute> forceCompute, Scalar deltaT, int numSubsteps, std::shared_ptr<ParticleData> particleData)
{
    this->m_force_compute = forceCompute;
    this->m_force_scaling_factor = 0.5 * (deltaT / numSubsteps);
    this->m_pdata = particleData;
}

//Destructor
RespaVelStep::~RespaVelStep() {}

// A function to execute a velocity "half" step according to the RESPA algorithm.
void RespaVelStep::executeStep(uint64_t timestep)
{
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);


    m_force_compute->compute(timestep);
    GlobalArray<Scalar>& forceArray = m_force_compute.getForceArray();

    for (unsigned int i = 0; i < m_pdata->getN(); i++) {
        h_vel.data[i].x = h_vel.data[i].x + m_force_scaling_factor * forceArray.x[i] / h_vel.data[i].w; //The "w" is the particle mass. For another example of this usage, see ParticleData::getMass
        h_vel.data[i].y = h_vel.data[i].y + m_force_scaling_factor * forceArray.y[i] / h_vel.data[i].w;
        h_vel.data[i].z = h_vel.data[i].z + m_force_scaling_factor * forceArray.z[i] / h_vel.data[i].z;
    }
}