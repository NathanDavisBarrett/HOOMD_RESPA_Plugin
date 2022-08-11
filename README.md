# Overview

The RESPA (Reversible Reference System Propagation Algorithm) integrator is intended to enable simulations to span different time and spatial scales. This is done by using multiple time steps for different forces within a system.

A typical velocity verlet integrator breaks each time step into three distinct sub-steps: A first-half velocity step, a position step, and a second-half velocity step.

1. $v_{i,t+0.5} = v_{i,t} + \frac{1}{2} \frac{F \left( x_{i,t} \right) }{m_i} \Delta t$
2. $x_{i,t+1} = x_{i,t} + v_{i,t+0.5} \Delta t$
3. $v_{i,t+1} = v_{i,t+0.5} + \frac{1}{2} \frac{F\left( x_{i,t+1} \right)}{m_i} \Delta t$

This pattern is used in the RESPA integrator but is used in a somewhat recursive fashion. The nominal time step (the time step used in the integrator's constructor call) represents one overarching step. But within each overarching step, each force can be "stepped" various times as shown in the following pseudo-code.

```
function executeForceStep(ThisForce):
  // Execute this force's first-half velocity step.

  if this force doesn't have a child force:
    // Execute a position step
  else:
    for each iteration of the child force:
      executeForceStep(childForce)

  // Execute this force's second-half velocity step.

for each overarching step:
  for each iteration of the parent-most force:
    executeForceStep(parentmostForce)
```

For example, if we only have two forces, F1 which is a "fast" force that requires a time step of 1e-5, and F2 which is a "slow" force that can handle a time step of 3e-5, we could specify the following configuration.

1. Overarching time step: 3e-5.
2. F2 executes once per overarching time step.
3. F1 execute three times per overarching time step.

Then the execution schedule would look something like this:

```
for each overarching step:
  Execute F2 first-half velocity step
    Execute F1 first-half velocity step
      Execute position step
    Execute F2 second-half velocity step

    Execute F1 first-half velocity step
      Execute position step
    Execute F2 second-half velocity step

    Execute F1 first-half velocity step
      Execute position step
    Execute F2 second-half velocity step
  Execute F2 second-half velocity step.
```

For more details and examples see these sources:

1. RESPA theory: http://www.columbia.edu/cu/chemistry/groups/berne/papers/jcp_97_1990_1992.pdf
2. OpenMM RESPA implementation: http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html

# About this plugin

[Hoomd-blue 2.9](https://github.com/glotzerlab/hoomd-blue/tree/2.9)

This plugin is based on hoomd-blue 2.9's example_plugin. For basics on how custom plugins should interact with hoomd-blue, see the documentation for the example_plugin.

Along the same lines, this plugin is intended for use with Hoomd-blue 2.9 and is not yet equipped to work with more recent versions of hoomd.

# Installation instructions

### Install Hoomd

To begin, make sure you have an installed and operational copy of hoomd-blue 2.9. If you do not have hoomd installed, clone the hoomd-blue github repository from the link above. Then checkout the branch entitled "2.9".

Open a terminal window in the cloned repository and execute the following commands:

```
mkdir build
cd build
cmake ..
cmake --build ./
cmake --install ./
```

You may need to install cmake, if you don't have it already. I suggest using Anaconda for the sake of simplicity: [Install cmake using Anaconda](https://anaconda.org/anaconda/cmake)

Compiling Hoomd for the first time can take around a half-hour. But you should only need to do this once.

### Install the plugin.

Clone this repository and open a terminal window in it. Execute the same commands to compile and install the plugin.

```
mkdir build
cd build
cmake ..
cmake --build ./
cmake --install ./
```

Note that sometimes the FindHOOMD script does not find the right hoomd directory. The cmake --install command should print out the directory that it will try to install the plugin to. Make sure that this aligns with where hoomd is actually installed to (printed during the Hoomd cmake --install command or by calling print(hoomd.__file__) in a python script).

# Usage instructions

The RESPA integrator can be used in a very similar fashion to the existing Hoomd integrators, with a couple of key differences. Instead of using Hoomd's automatic detection and addition of ForceCompute objects to the integrator's list of forces, you must add each force explicitly. This is because each force not only needs to be added, but it must also have an accompanying compute frequency. An example of how to add forces to the RESPA integrator is as follows.

!@#$!@#$!@#$!@$##@!$#@! CHECK THIS FUNCTION CALL:
```
integrator = respa_plugin.mode_respa(timestep,[
  [force1, 3],
  [force2, 1]
  ])
```

The number included with each force is the compute frequency. In other words, that particular force will be computed that many times per overarching time step. Here is a good point to point out that each force's compute frequency must be a multiple of the next-greatest frequency (e.g. 3 is a multiple of 1. If I were to add another force it would need a frequency of 3, 6, 9, 12, etc.).
