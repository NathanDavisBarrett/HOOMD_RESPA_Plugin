//
// Created by nathan on 8/12/21.
//

#include "MultipleTimestepIntegrator.h"
#ifdef ENABLE_HIP
#include "MultipleTimestepIntegrator.cuh"
#endif

#include "RespaStep.h"
#include "RespaVelStep.h"
#include "RespaPosStep.h"

/*! \file MultipleTimestepIntegrator.cc
    \brief Definition of MultipleTimestepIntegrator
*/

// ********************************
// here follows the code for MultipleTimestepIntegrator on the CPU

/*! \param sysdef System to zero the velocities of
 */
MultipleTimestepIntegrator::MultipleTimestepIntegrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
: Integrator(sysdef, deltaT), m_prepared(false), m_aniso_mode(Automatic) {
    m_exec_conf->msg->notice(5) << "Constructing MultipleTimestepIntegrator" << endl;
}

MultipleTimestepIntegrator::~MultipleTimestepIntegrator() {
    m_exec_conf->msg->notice(5) << "Destroying MultipleTimestepIntegrator" << endl;

    for (unsigned int i = 0; i < m_respa_steps.size(); i++) {
        delete m_respa_steps.at(i);
    }

#ifdef ENABLE_MPI
    if (m_comm) {
        m_comm->getComputeCallbackSignal()
        .disconnect<MultipleTimestepIntegrator, &MultipleTimestepIntegrator::updateRigidBodies>(this);
    }
#endif
}

/*! \param prof The profiler to set
    Sets the profiler both for this class
*/
void MultipleTimestepIntegrator::setProfiler(std::shared_ptr<Profiler> prof)
{
    Integrator::setProfiler(prof);
}

/*! Create the substeps needed for each loop and subloop in the RESPA algorithm.
*/
void MultipleTimestepIntegrator::createSubsteps(vector<std::pair<std::shared_ptr<ForceCompute>, int>> forceGroups, int parentSubsteps) {
    std::pair<std::shared_ptr<ForceCompute>, int> topGroup = forceGroups.at(0);

    std::shared_ptr<ForceCompute> topForce = topGroup.first;
    int topSubsteps = topGroup.second;

    if ((topSubsteps % parentSubsteps) != 0) {
        //Error, The nubmer of substeps for each RESPA group must be a multiple of the number for the previous group.
        throw std::invalid_argument("The nubmer of substeps for each RESPA group must be a multiple of the number for the previous group.");
    }

    int stepsPerParentStep = topSubsteps / parentSubsteps;

    for (int i = 0; i < stepsPerParentStep; i++) {
        m_respa_steps.push_back(new RespaVelStep(topForce, m_deltaT, topSubsteps, m_pdata));

        if (forceGroups.size() == 1) { //Meaning this is the inner-most forcegroup.
            m_respa_steps.push_back(new RespaPosStep(m_deltaT, topSubsteps, m_pdata));
        }
        else {
            vector<std::pair<std::shared_ptr<ForceCompute>, int>> tempGroups = forceGroups;

            tempGroups.erase(tempGroups.begin());

            MultipleTimestepIntegrator::createSubsteps(tempGroups, topSubsteps);
        }

        m_respa_steps.push_back(new RespaVelStep(topForce, m_deltaT, topSubsteps, m_pdata));
    }
}

/*! Prepare for the run.
*/
void MultipleTimestepIntegrator::prepRun(uint64_t timestep) {
    //First, make sure the vector of ForceComputes are organized to put the least frequent force at the front, and the most frequent force at the back.
    struct SortHelper {
        inline bool operator() (const std::pair<std::shared_ptr<ForceCompute>, int> pair1, const std::pair<std::shared_ptr<ForceCompute>, int> pair2) {
            return (pair1.second < pair2.second);
        }
    }

    std::sort(m_respa_forces.begin(), m_respa_forces.end(), SortHelper());

    //Now create the substeps needed to execute the RESPA algorithm.
    MultipleTimestepIntegrator::createSubsteps(m_respa_forces, 1);

    m_prepared = true;
}

/*! Perform the needed calculations according to the RESPA algorithm
    \param timestep Current time step of the simulation (i.e. the timestep number??)
*/
void MultipleTimestepIntegrator::update(uint64_t timestep)
{
    Integrator::update(timestep);

    // ensure that prepRun() has been called
    assert(m_prepared);

    if (m_prof)
        m_prof->push("MultipleTimestepIntegrator");

    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    // IMPLEMENT RESPA ALGORITHM HERE.
    // TODO:
    // Make the initializer function to set up the pattern at which to execute the steps.
    // Learn how to access and execute ForceComputes. (Do a search algorithm on ForceCompute.h, see where it's used and how to determine what kind of data object a "force" (such as LJ) is.)
    //     Use force->compute(timestep); See Integrator.cc, computeNetForce() for an example.
    //     But how do you add a new force!? I'm gonna need to override that to mandate a frequency along with the force.

    for (unsigned int i = 0; i < m_respa_steps; i++) {
        m_respa_steps.at(i)->executeStep(timestep);
    }


    if (m_prof)
        m_prof->pop();
}

/*! \param deltaT new deltaT to set
*/
void MultipleTimestepIntegrator::setDeltaT(Scalar deltaT)
{
    Integrator::setDeltaT(deltaT);

    if (m_rigid_bodies)
    {
        m_rigid_bodies->setDeltaT(deltaT);
    }
}

/* Add a new force/frequency pair to the integrator.
 *
 */
void MultipleTimestepIntegrator::addForce(std::shared_ptr<ForceCompute> force, int frequency) {
    std::pair<std::shared_ptr<ForceCompute>, int> newForce;
    newForce.first = force;
    newForce.second = frequency;
    m_respa_forces.push_back(newForce);
}

/* Export the CPU Integrator to be visible in the python module
 */
void export_MultipleTimestepIntegrator(pybind11::module& m)
{
    pybind11::class_<MultipleTimestepIntegrator, Integrator, std::shared_ptr<MultipleTimestepIntegrator>>(m, "MultipleTimestepIntegrator")
    .def(py::init<std::shared_ptr<SystemDefinition>, Scalar>())
    .def("addForce", &MultipleTimestepIntegrator::addForce);
}

// ********************************
// here follows the code for MultipleTimestepIntegrator on the GPU

#ifdef ENABLE_HIP

/*! \param sysdef System to zero the velocities of
 */
MultipleTimestepIntegratorGPU::MultipleTimestepIntegratorGPU(std::shared_ptr<SystemDefinition> sysdef)
: MultipleTimestepIntegrator(sysdef)
{
}

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void MultipleTimestepIntegratorGPU::update(uint64_t timestep)
{
    Integrator::update(timestep);
    if (m_prof)
        m_prof->push("MultipleTimestepIntegrator");

    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);

    // call the kernel defined in MultipleTimestepIntegrator.cu
    gpu_zero_velocities(d_vel.data, m_pdata->getN());

    // check for error codes from the GPU if error checking is enabled
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop();
}

/* Export the GPU Integrator to be visible in the python module
 */
void export_MultipleTimestepIntegratorGPU(pybind11::module& m)
{
    pybind11::class_<MultipleTimestepIntegratorGPU, MultipleTimestepIntegrator, std::shared_ptr<MultipleTimestepIntegratorGPU>>(
            m,
            "MultipleTimestepIntegratorGPU")
            .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
}

#endif // ENABLE_HIP
