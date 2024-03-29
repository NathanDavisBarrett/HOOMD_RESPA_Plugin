#include "RespaIntegrator.h"

#include <string>
#include <sstream>
#include <map>
#include <cmath>

namespace py = pybind11;

#ifdef ENABLE_HIP
#include "RespaIntegrator.cuh"
#endif

/*! \param sysdef The system to associate this integrator with.
    \param deltaT The overarching timestep value to use.
 */
RespaIntegrator::RespaIntegrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
: Integrator(sysdef, deltaT), m_prepared(false), m_aniso_mode(Automatic) {
    m_exec_conf->msg->notice(5) << "Constructing RespaIntegrator" << std::endl;
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

/*! Calculates the force scaling factor for a Respa "velocity step" with the given number of substeps.
    \param numSubsteps The number of substeps that occur between each iteration of this substep.
 */
Scalar RespaIntegrator::calculateForceScalingFactor(int numSubsteps) {
    return 0.5 * (m_deltaT / numSubsteps);
}

/*! Calculates the velocity scaling factor for a Respa "position step" with the given number of substeps.
    \param numSubsteps The number of substeps that occur between each iteration of this substep.
 */
Scalar RespaIntegrator::calculateVelScalingFactor(int numSubsteps) {
    return (m_deltaT / numSubsteps);
}

/*! Adds the appropriate substep information to the m_respa_step_types, m_respa_step_force_computes, m_respa_step_force_scaling_factors, and m_respa_step_vel_scaling_factors vectors for a given substep.
    \param stepType The identifier for which type of substep this particular substep is.
    \param forceCompute The ForceCompute associated with this substep (NULL if this substep does not have an associated ForceCompute)
    \param numSubsteps The number of substeps that will occur within each iteration of this substep.
 */
void RespaIntegrator::addSubstep(int stepType, std::shared_ptr<ForceCompute> forceCompute, int numSubsteps) {
    Scalar forceScalingFactor = 0;
    Scalar velScalingFactor = 0;

    if (stepType == POS_STEP) {
        if (forceCompute != NULL) {
            throw std::invalid_argument("forceCompute must be null in order to specify a PosStep");
        }
        velScalingFactor = calculateVelScalingFactor(numSubsteps);
    }
    else if ((stepType == VEL_STEP_1) || (stepType == VEL_STEP_2)) {
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
    \param forceGroups A vector for ForceCompute-Frequency pairs. Each frequency in this case is how many times each substep will execute per iteration of it's parent substep.
    \parentSubsteps The number of times this substep's parent substep is executed per overaching timestep.
*/
void RespaIntegrator::createSubsteps(std::vector<std::pair<std::shared_ptr<ForceCompute>, int>> forceGroups, int parentSubsteps) {
    std::pair<std::shared_ptr<ForceCompute>, int> topGroup = forceGroups.at(0);

    std::shared_ptr<ForceCompute> topForce = topGroup.first;
    int topSubsteps = topGroup.second;

    if ((topSubsteps % parentSubsteps) != 0) {
        throw std::invalid_argument("The nubmer of substeps for each RESPA group must be a multiple of the number for the previous group.");
    }

    int stepsPerParentStep = topSubsteps / parentSubsteps;

    for (int i = 0; i < stepsPerParentStep; i++) {
        this->addSubstep(VEL_STEP_1,topForce,topSubsteps);

        if (forceGroups.size() == 1) { //Meaning this is the inner-most forcegroup.
            this->addSubstep(POS_STEP, NULL, topSubsteps);
        }
        else {
            std::vector<std::pair<std::shared_ptr<ForceCompute>, int>> tempGroups = forceGroups;

            tempGroups.erase(tempGroups.begin());

            RespaIntegrator::createSubsteps(tempGroups, topSubsteps);
        }
        this->addSubstep(VEL_STEP_2,topForce,topSubsteps);
    }
}

/*! Prepare for the run by setting up the respa sub-step schedule and setting m_prepared to true.
    \param timestep The number correspinding this the current overaching step in the simulation. This should almost allways be zero in this case since, in almost every simlatuion, the simulation starts at timestep number zero.
*/
void RespaIntegrator::prepRun(unsigned int timestep) {
    this->Integrator::computeNetForce(timestep);
    //First, make sure the vector of ForceComputes are organized to put the least frequent force at the front, and the most frequent force at the back.
    struct SortHelper {
        inline bool operator() (const std::pair<std::shared_ptr<ForceCompute>, int> pair1, const std::pair<std::shared_ptr<ForceCompute>, int> pair2) {
            return (pair1.second < pair2.second);
        }
    };

    std::sort(m_respa_forces.begin(), m_respa_forces.end(), SortHelper());

    //Now create the substeps needed to execute the RESPA algorithm.
    RespaIntegrator::createSubsteps(m_respa_forces, 1);

    if (m_forces.size() == 0) {
        m_forces.reserve(m_respa_forces.size());
        for (unsigned int i = 0; i < m_respa_forces.size(); i++) {
            m_forces.push_back(m_respa_forces.at(i).first);
        }
    }

    m_prepared = true;

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
}

/*! Perform the needed calculations according to the RESPA algorithm
    \param timestep The iteration number (of overarching timesteps) correspinding to this particular iteration of updates.
*/
void RespaIntegrator::update(unsigned int timestep)
{
    // ensure that prepRun() has been called
    assert(m_prepared);

    if (m_prof)
        m_prof->push("RespaIntegrator");

    // access the particle data for writing on the CPU
    assert(m_pdata);

    for (unsigned int i = 0; i < m_respa_step_types.size(); i++) {
        int stepType = m_respa_step_types.at(i);

        if ((stepType == VEL_STEP_1) || (stepType == VEL_STEP_2)) {
            std::shared_ptr<ForceCompute> forceCompute = m_respa_step_force_computes.at(i);
            Scalar forceScalingFactor = m_respa_step_force_scaling_factors.at(i);

            int targTimestep = timestep;
            if (stepType == VEL_STEP_2) {
                targTimestep++;
            }
            forceCompute->compute(targTimestep); //Should this only happen on the second-half step? Or does HOOMD know not to re-compute if it's been computed already.

            ArrayHandle<Scalar4>  h_force(forceCompute->getForceArray(),
                                          access_location::host,
                                          access_mode::read);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                          access_location::host,
                                          access_mode::readwrite);

            for (unsigned int j = 0; j < m_pdata->getN(); j++) {
                Scalar forceX = h_force.data[j].x;
                Scalar forceY = h_force.data[j].y;
                Scalar forceZ = h_force.data[j].z;


                h_vel.data[j].x = h_vel.data[j].x + forceScalingFactor * forceX / h_vel.data[j].w; //The "w" is the particle mass. For another example of this usage, see ParticleData::getMass
                h_vel.data[j].y = h_vel.data[j].y + forceScalingFactor * forceY / h_vel.data[j].w;
                h_vel.data[j].z = h_vel.data[j].z + forceScalingFactor * forceZ / h_vel.data[j].w;
            }
        }
        else if (stepType == POS_STEP) {
            Scalar velScalingFactor = m_respa_step_vel_scaling_factors.at(i);

            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                       access_location::host,
                                       access_mode::readwrite);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);

            const BoxDim& box = m_pdata->getGlobalBox();

            for (unsigned int j = 0; j < m_pdata->getN(); j++)
            {
                Scalar3 pos = make_scalar3(
                    h_pos.data[j].x + velScalingFactor * h_vel.data[j].x,
                    h_pos.data[j].y + velScalingFactor * h_vel.data[j].y,
                    h_pos.data[j].z + velScalingFactor * h_vel.data[j].z);

                pos = box.minImage(pos);

                h_pos.data[j].x = pos.x;
                h_pos.data[j].y = pos.y;
                h_pos.data[j].z = pos.z;
            }
        }
        else {
            throw std::invalid_argument(std::to_string(stepType) + " is not a valid stepType");
        }
    }

    this->Integrator::computeNetForce(timestep);

    if (m_prof)
        m_prof->pop();
}

/*! Sets the value of "deltaT", the duration of the overarching timestep.
    \param deltaT new deltaT to set
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

/*! Add a new force/frequency pair to the integrator.
    \param force The ForceCompute object to be added.
    \param frequency The number of times per overarching timestep that this force should be computed.
 */
void RespaIntegrator::addForce(std::shared_ptr<ForceCompute> force, int frequency) {
    //m_exec_conf->msg->warning() << "addForce called" << std::endl;
    std::pair<std::shared_ptr<ForceCompute>, int> newForce;
    newForce.first = force;
    newForce.second = frequency;
    m_respa_forces.push_back(newForce);
}

/*! Prints the schedule of respa sub-steps that this integrator will run.
 */
void RespaIntegrator::printSchedule() {
    if (!m_prepared) {
        m_exec_conf->msg->warning() << "printSchedule() is being called on an unprepared integrator. The following schedule might be inaccurate.\n";
    }

    std::stringstream ss;

    std::map< std::string , char> ptrNames;
    char maxChar = 'A';

    for (size_t i = 0; i < m_respa_step_types.size(); i++) {
        int stepType = m_respa_step_types.at(i);
        if (stepType == VEL_STEP_1) {
            ss << "Vel Step 1 ";
        }
        else if (stepType == VEL_STEP_2) {
            ss << "Vel Step 2 ";
        }
        else {
            ss << "Pos Step\n";
            continue;
        }

        std::stringstream ptrSS;
        ptrSS << m_respa_step_force_computes.at(i);

        std::string forceComputePointerName = ptrSS.str();

        auto it = ptrNames.find(forceComputePointerName);
        if (it == ptrNames.end()) {
            ptrNames[forceComputePointerName] = maxChar;
            maxChar++;
        }

        char targChar = ptrNames[forceComputePointerName];

        ss << "FC_" << targChar << "\n";
    }

    py::print(ss.str());
}

/* Export the CPU Integrator to be visible in the python module
 */
void export_RespaIntegrator(pybind11::module& m)
{
    pybind11::class_<RespaIntegrator, Integrator, std::shared_ptr<RespaIntegrator>>(m, "RespaIntegrator")
    .def(py::init<std::shared_ptr<SystemDefinition>, Scalar>())
    .def("getNDOF", &RespaIntegrator::getNDOF)
    .def("getRotationalNDOF", &RespaIntegrator::getRotationalNDOF)
    .def("addForce", &RespaIntegrator::addForce)
    .def("printSchedule", &RespaIntegrator::printSchedule);
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
