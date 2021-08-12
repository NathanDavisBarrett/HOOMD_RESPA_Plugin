//
// Created by nathan on 8/12/21.
//

#ifndef MULTIPLETIMESTEPPLUGIN_RESPAVELSTEP_H
#define MULTIPLETIMESTEPPLUGIN_RESPAVELSTEP_H

#include "RespaStep.h"

class RespaVelStep : public RespaStep
        {
        private:
            std::shared_ptr<ForceCompute> m_force_compute;
            Scalar m_force_scaling_factor;

        public:
            // Constructor
            RespaVelStep(std::shared_ptr<ForceCompute>, Scalar, int, std::shared_ptr<ParticleData>);

            //Destructor
            ~RespaVelStep();

            // A function to execute a velocity "half" step according to the RESPA algorithm.
            void executeStep(uint64_t timestep);

        };

#endif //MULTIPLETIMESTEPPLUGIN_RESPAVELSTEP_H
