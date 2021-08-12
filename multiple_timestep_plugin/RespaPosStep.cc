//
// Created by nathan on 8/12/21.
//

#include "RespaPosStep.h"

// Constructor
RespaPosStep::RespaPosStep(Scalar deltaT, int numSubsteps, std::shared_ptr<ParticleData> particleData)
{
    this->m_vel_scaling_factor = deltaT / numSubsteps;
    this->m_pdata = particleData;
}

//Destructor
RespaPosStep::~RespaPosStep() {}

// A function to execute a position "half" step according to the RESPA algorithm.
void RespaPosStep::executeStep(uint64_t timestep)
{
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);


    for (unsigned int i = 0; i < m_pdata->getN(); i++)
    {
        h_pos.data[i].x = h_pos.data[i].x + m_vel_scaling_factor * h_vel.data[i].x;
        h_pos.data[i].y = h_pos.data[i].y + m_vel_scaling_factor * h_vel.data[i].y;
        h_pos.data[i].z = h_pos.data[i].z + m_vel_scaling_factor * h_vel.data[i].z;
    }
}