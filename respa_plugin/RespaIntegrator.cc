//
// Created by nathan on 8/12/21.
//

#include "RespaIntegrator.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

namespace py = pybind11;

#ifdef ENABLE_HIP
#include "RespaIntegrator.cuh"
#endif

/*! \file RespaIntegrator.cc
    \brief Definition of RespaIntegrator
*/

// ********************************
// here follows the code for RespaIntegrator on the CPU

/*! \param sysdef System to zero the velocities of
 */
RespaIntegrator::RespaIntegrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
: Integrator(sysdef, deltaT), m_prepared(false), m_aniso_mode(Automatic) {
    m_exec_conf->msg->notice(5) << "Constructing RespaIntegrator" << std::endl;
    //m_exec_conf->msg->warning() << "Constructing RespaIntegrator:" << this << std::endl;
}

RespaIntegrator::~RespaIntegrator() {
    m_exec_conf->msg->notice(5) << "Destroying RespaIntegrator" << std::endl;

#ifdef ENABLE_MPI
    if (m_comm) {
        m_comm->getComputeCallbackSignal()
        .disconnect<RespaIntegrator, &RespaIntegrator::updateRigidBodies>(this);
    }
#endif
}

/*! \param prof The profiler to set
    Sets the profiler both for this class
*/
void RespaIntegrator::setProfiler(std::shared_ptr<Profiler> prof)
{
    Integrator::setProfiler(prof);
}

Scalar RespaIntegrator::calculateForceScalingFactor(int numSubsteps) {
    return 0.5 * (m_deltaT / numSubsteps);
}

Scalar RespaIntegrator::calculateVelScalingFactor(int numSubsteps) {
    return (m_deltaT / numSubsteps);
}

void RespaIntegrator::addSubstep(int stepType, std::shared_ptr<RespaForceCompute> forceCompute, int numSubsteps) {
    Scalar forceScalingFactor = NULL;
    Scalar velScalingFactor = NULL;

    if (stepType == POS_STEP) {
        if (forceCompute != NULL) {
            throw std::invalid_argument("forceCompute must be null in order to specify a PosStep");
        }
        velScalingFactor = calculateVelScalingFactor(numSubsteps);
    }
    else if (stepType == VEL_STEP) {
        forceScalingFactor = calculateForceScalingFactor(numSubsteps);
    }
    else {
        throw std::invalid_argument(std::to_string(stepType) + " is not a valid stepType");
    }

    this->m_respa_step_types.push_back(stepType);
    this->m_respa_step_force_computes.push_back(forceCompute);
    this->m_respa_step_force_scaling_factors.push_back(forceScalingFactor);
    this->m_respa_step_vel_scaling_factors.push_back(velScalingFactor);
}

/*! Create the substeps needed for each loop and subloop in the RESPA algorithm.
*/
void RespaIntegrator::createSubsteps(std::vector<std::pair<std::shared_ptr<RespaForceCompute>, int>> forceGroups, int parentSubsteps) {
    //m_exec_conf->msg->warning() << "createSubsteps called" << std::endl;

    std::pair<std::shared_ptr<RespaForceCompute>, int> topGroup = forceGroups.at(0);

    std::shared_ptr<RespaForceCompute> topForce = topGroup.first;
    int topSubsteps = topGroup.second;

    if ((topSubsteps % parentSubsteps) != 0) {
        //Error, The nubmer of substeps for each RESPA group must be a multiple of the number for the previous group.
        throw std::invalid_argument("The nubmer of substeps for each RESPA group must be a multiple of the number for the previous group.");
    }

    int stepsPerParentStep = topSubsteps / parentSubsteps;

    for (int i = 0; i < stepsPerParentStep; i++) {
        this->addSubstep(VEL_STEP,topForce,topSubsteps);

        if (forceGroups.size() == 1) { //Meaning this is the inner-most forcegroup.
            this->addSubstep(POS_STEP, NULL, topSubsteps);
        }
        else {
            std::vector<std::pair<std::shared_ptr<RespaForceCompute>, int>> tempGroups = forceGroups;

            tempGroups.erase(tempGroups.begin());

            RespaIntegrator::createSubsteps(tempGroups, topSubsteps);
        }
        this->addSubstep(VEL_STEP,topForce,topSubsteps);
    }
}

/*! Prepare for the run.
*/
void RespaIntegrator::prepRun(unsigned int timestep) {
    //m_exec_conf->msg->warning() << "RespaIntegrator prepRun called" << std::endl;

    //First, make sure the vector of ForceComputes are organized to put the least frequent force at the front, and the most frequent force at the back.
    struct SortHelper {
        inline bool operator() (const std::pair<std::shared_ptr<RespaForceCompute>, int> pair1, const std::pair<std::shared_ptr<RespaForceCompute>, int> pair2) {
            return (pair1.second < pair2.second);
        }
    };

    std::sort(m_respa_forces.begin(), m_respa_forces.end(), SortHelper());

    //Now create the substeps needed to execute the RESPA algorithm.
    RespaIntegrator::createSubsteps(m_respa_forces, 1);

    m_prepared = true;

    //m_exec_conf->msg->warning() << "Initial Positions/Velocities:" << std::endl;

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    // for (unsigned int i = 0; i < m_pdata->getN(); i++) {
    //     //m_exec_conf->msg->warning() << i << " x: " << h_pos.data[i].x << " y: " << h_pos.data[i].y << " z: " << h_pos.data[i].z << " vx: " << h_vel.data[i].x << " vy: " << h_vel.data[i].y << " vz: " << h_vel.data[i].z << "\n";
    // }

}

/*! Perform the needed calculations according to the RESPA algorithm
    \param timestep Current time step of the simulation (i.e. the timestep number??)
*/
void RespaIntegrator::update(unsigned int timestep)
{
    Integrator::update(timestep);
    //m_exec_conf->msg->warning() << "TIMESTEP: " << timestep << std::endl;

    // ensure that prepRun() has been called
    assert(m_prepared);

    if (m_prof)
        m_prof->push("RespaIntegrator");

    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    //m_exec_conf->msg->warning() << "\tThere are " << m_respa_step_types.size() << " respa steps to execute." << std::endl;

    for (unsigned int i = 0; i < m_respa_step_types.size(); i++) {
        //m_exec_conf->msg->warning() << "\tRespaStep #" << i << ":" << std::endl;

        //m_exec_conf->msg->warning() << "\t\tPositions/Velocities:" << "\n";

//        for (unsigned int i = 0; i < m_pdata->getN(); i++) {
//            m_exec_conf->msg->warning() << "\t\t\t" << i << " x: " << h_pos.data[i].x << " y: " << h_pos.data[i].y << " z: " << h_pos.data[i].z << " vx: " << h_vel.data[i].x << " vy: " << h_vel.data[i].y << " vz: " << h_vel.data[i].z << "\n";
//        }

        int stepType = m_respa_step_types.at(i);
        if (stepType == VEL_STEP) {
            //m_exec_conf->msg->warning() << "\t\tVEL_STEP" << std::endl;
            std::shared_ptr<RespaForceCompute> forceCompute = m_respa_step_force_computes.at(i);
            Scalar forceScalingFactor = m_respa_step_force_scaling_factors.at(i);

            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                       access_location::host,
                                       access_mode::readwrite);

            // m_exec_conf->msg->warning() << "########### CALLING COMPUTE. ###########\n";
            // forceCompute->compute(timestep);
            // m_exec_conf->msg->warning() << "########### COMPUTE CALL COMPLETE. ###########\n";

            ArrayHandle<Scalar4>  h_force(forceCompute->getForceArray(),
                                          access_location::host,
                                          access_mode::read);

            Scalar minForce = NULL;
            Scalar maxForce = NULL;
            Scalar avgForce = 0.0;


            for (unsigned int i = 0; i < m_pdata->getN(); i++) {
                Scalar forceX = h_force.data[i].x;
                Scalar forceY = h_force.data[i].y;
                Scalar forceZ = h_force.data[i].z;

                if (i % 100 == 0) {
                    //m_exec_conf->msg->warning() << " Fx:" << forceX << " Fy:" << forceY << " Fz:" << forceZ << "\n";
                }

                Scalar forceMag = pow((double)(forceX*forceX + forceY*forceY + forceZ*forceZ),0.5);

                if (maxForce == NULL) {
                    maxForce = forceMag;
                }
                else if (forceMag > maxForce) {
                    maxForce = forceMag;
                }

                if (minForce == NULL) {
                    minForce = forceMag;
                }
                else if (forceMag < minForce) {
                    minForce = forceMag;
                }

                avgForce = ((avgForce * (double)i) + forceMag) / ((double)(i+1));

                h_vel.data[i].x = h_vel.data[i].x + forceScalingFactor * forceX / h_vel.data[i].w; //The "w" is the particle mass. For another example of this usage, see ParticleData::getMass
                h_vel.data[i].y = h_vel.data[i].y + forceScalingFactor * forceY / h_vel.data[i].w;
                h_vel.data[i].z = h_vel.data[i].z + forceScalingFactor * forceZ / h_vel.data[i].w;
            }

            //m_exec_conf->msg->warning() << "\t\tminForce: " << minForce << std::endl;
            //m_exec_conf->msg->warning() << "\t\tmaxForce: " << maxForce << std::endl;
            //m_exec_conf->msg->warning() << "\t\tavgForce: " << avgForce << std::endl;
        }
        else if (stepType == POS_STEP) {
            //m_exec_conf->msg->warning() << "\t\tPOS_STEP" << std::endl;
            Scalar velScalingFactor = m_respa_step_vel_scaling_factors.at(i);

            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                       access_location::host,
                                       access_mode::readwrite);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);

            Scalar maxD = NULL;
            Scalar minD = NULL;
            Scalar avgD = NULL;

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
                Scalar dx = velScalingFactor * h_vel.data[i].x;
                Scalar dy = velScalingFactor * h_vel.data[i].y;
                Scalar dz = velScalingFactor * h_vel.data[i].z;

                Scalar d = pow((double)(dx*dx + dy*dy + dz*dz),0.5);

                if (maxD == NULL) {
                    maxD = d;
                }
                else if (d > maxD){
                    maxD = d;
                }

                if (minD == NULL) {
                    minD = d;
                }
                else if (d < minD) {
                    minD = d;
                }

                avgD = (avgD*((double)i) + d) / ((double)(i+1));

                h_pos.data[i].x = h_pos.data[i].x + velScalingFactor * h_vel.data[i].x;
                h_pos.data[i].y = h_pos.data[i].y + velScalingFactor * h_vel.data[i].y;
                h_pos.data[i].z = h_pos.data[i].z + velScalingFactor * h_vel.data[i].z;
            }

            //m_exec_conf->msg->warning() << "\t\tminD: " << minD << std::endl;
            //m_exec_conf->msg->warning() << "\t\tmaxD: " << maxD << std::endl;
            //m_exec_conf->msg->warning() << "\t\tavgD: " << avgD << std::endl;
        }
        else {
            throw std::invalid_argument(std::to_string(stepType) + " is not a valid stepType");
        }
    }


    if (m_prof)
        m_prof->pop();
}

/*! \param deltaT new deltaT to set
*/
void RespaIntegrator::setDeltaT(Scalar deltaT)
{
    Integrator::setDeltaT(deltaT);
}

/*! Get the number of degrees of freedom granted to a given group
*/
unsigned int RespaIntegrator::getNDOF(std::shared_ptr<ParticleGroup> query_group) {
    unsigned int group_size = query_group->getNumMembersGlobal();

    return m_sysdef->getNDimensions() * group_size;
}

unsigned int RespaIntegrator::getRotationalNDOF(std::shared_ptr<ParticleGroup> group)
{
    int res = 0;

    bool aniso = false;

    // This is called before prepRun, so we need to determine the anisotropic modes independently here.
    // It cannot be done earlier.
    // set (an-)isotropic integration mode
    switch (m_aniso_mode)
    {
        case Anisotropic:
            aniso = true;
            break;
            case Automatic:
                default:
                    aniso = getAnisotropicMode();
                    break;
    }

    //m_exec_conf->msg->notice(8) << "RespaIntegrator: Setting anisotropic mode = " << aniso << std::endl;

    if (aniso)
    {
        unsigned int group_size = group->getNumMembers();
        unsigned int group_dof = 0;
        unsigned int dimension = m_sysdef->getNDimensions();
        unsigned int dof_one;
        ArrayHandle<Scalar3> h_moment_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++) {
            unsigned int j = group->getMemberIndex(group_idx);
            if (dimension == 3) {
                dof_one = 3;
                if (fabs(h_moment_inertia.data[j].x) < EPSILON) {
                    dof_one--;
                }
                if (fabs(h_moment_inertia.data[j].y) < EPSILON) {
                    dof_one--;
                }
                if (fabs(h_moment_inertia.data[j].z) < EPSILON) {
                    dof_one--;
                }
            }
            else {
                dof_one = 1;
                if (fabs(h_moment_inertia.data[j].z) < EPSILON) {
                    dof_one--;
                }
            }
            group_dof += dof_one;
        }
#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition()) {
            MPI_Allreduce(MPI_IN_PLACE, &group_dof, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif
        res += group_dof;
    }

    return res;
}

/*! Set the anisotropic mode of the integrator
 */
void RespaIntegrator::setAnisotropicMode(AnisotropicMode mode) {
    m_aniso_mode = mode;
}

/*! get the anisotropic mode of the integrator
 */
bool RespaIntegrator::getAnisotropicMode() {
    return m_aniso_mode;
}

/* Add a new force/frequency pair to the integrator.
 *
 */
void RespaIntegrator::addForce(std::shared_ptr<RespaForceCompute> force, int frequency) {
    //m_exec_conf->msg->warning() << "addForce called" << std::endl;
    std::pair<std::shared_ptr<RespaForceCompute>, int> newForce;
    newForce.first = force;
    newForce.second = frequency;
    m_respa_forces.push_back(newForce);
}

/* Export the CPU Integrator to be visible in the python module
 */
void export_RespaIntegrator(pybind11::module& m)
{
    pybind11::class_<RespaIntegrator, Integrator, std::shared_ptr<RespaIntegrator>>(m, "RespaIntegrator")
    .def(py::init<std::shared_ptr<SystemDefinition>, Scalar>())
    .def("getNDOF", &RespaIntegrator::getNDOF)
    .def("getRotationalNDOF", &RespaIntegrator::getRotationalNDOF)
    .def("addForce", &RespaIntegrator::addForce);
}

// ********************************
// here follows the code for RespaIntegrator on the GPU

#ifdef ENABLE_HIP

/*! \param sysdef System to zero the velocities of
 */
RespaIntegratorGPU::RespaIntegratorGPU(std::shared_ptr<SystemDefinition> sysdef)
: RespaIntegrator(sysdef)
{
}

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void RespaIntegratorGPU::update(uint64_t timestep)
{
    Integrator::update(timestep);
    if (m_prof)
        m_prof->push("RespaIntegrator");

    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);

    // call the kernel defined in RespaIntegrator.cu
    gpu_zero_velocities(d_vel.data, m_pdata->getN());

    // check for error codes from the GPU if error checking is enabled
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop();
}

/* Export the GPU Integrator to be visible in the python module
 */
void export_RespaIntegratorGPU(pybind11::module& m)
{
    pybind11::class_<RespaIntegratorGPU, RespaIntegrator, std::shared_ptr<RespaIntegratorGPU>>(
            m,
            "RespaIntegratorGPU")
            .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
}

#endif // ENABLE_HIP
