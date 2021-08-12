//
// Created by nathan on 8/12/21.
//

#ifndef MULTIPLETIMESTEPPLUGIN_RESPAPOSSTEP_H
#define MULTIPLETIMESTEPPLUGIN_RESPAPOSSTEP_H

class RespaPosStep : public RespaStep
        {
        private:
            Scalar m_vel_scaling_factor;

        public:
            // Constructor
            RespaPosStep(Scalar, int, std::shared_ptr<ParticleData>);

            //Destructor
            ~RespaPosStep();

            // A function to execute a position "half" step according to the RESPA algorithm.
            void executeStep(uint64_t timestep);
        };

#endif //MULTIPLETIMESTEPPLUGIN_RESPAPOSSTEP_H
