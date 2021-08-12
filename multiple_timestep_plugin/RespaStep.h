//
// Created by nathan on 8/12/21.
//

#ifndef MULTIPLETIMESTEPPLUGIN_RESPASTEP_H
#define MULTIPLETIMESTEPPLUGIN_RESPASTEP_H

class RespaStep
{
private:
    std::shared_ptr<ParticleData> m_pdata;

public:
    // A function to execute the step. Details on what things to do in this step will be specified in daughter classes.
    virutal void executeStep(uint64_t timestep);
};

#endif //MULTIPLETIMESTEPPLUGIN_RESPASTEP_H
