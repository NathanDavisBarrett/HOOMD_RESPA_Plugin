R""" Respa Pair potentials

This is the adaptation of the hoomd.md.pair classes for the Respa Integrator.

A few notes, for simplicity's sake, the RespaParPotential classes are kept virtaully identical to the normal
PairPotential class. Ths only difference is that they operate on RespaForceCompute's version of computeForces.

Thus, they will require a particle group as an argument in addition to the normal pair potential arguments.
Similarly, the ForceCompute will only calculate forces on the particles in the specified group. Nonetheless, you
will still need to provide pair coefficients for all particles in the system, regardless of whether or not they
are in the specified group (since it still operates practically entirely on the original pair potential's methods).
Since coefficients for particle types not used in the specified group will never be used, I recomend you set them to
a very large number. That way, if they are called somehow, the system will immediately become unstable and you will have
an indication that something has gone wrong.

"""

# TO-DO:
# * Investigate whether or not you can just override self.cpp_force to be a RespaForceCompute object, or how that even works for the normal pair class.
#      Other wise, we might not be able to inherit from the normal pair class. Yikes.

# NOTE: As seen on the __init__ function for pair.lj, self.cpp_force and self.cpp_class are both set in the child class. So we're good. PFEW!

print("respa_pair file.")
from hoomd.md.pair import pair
from hoomd.md import _md
import hoomd
from hoomd.md.pair import lj

print("importing plugin.")
from hoomd.multiple_timestep_plugin import _multiple_timestep_plugin


class respa_pair(pair):
    R""" Respa pair potential documentation.
    This class simply extends the hoomd.md.pair.pair class to act as a RespaPotentialPair
    """

    def __init__(self, r_cut, nlist, group, name=None):
        hoomd.util.print_status_line();

        pair.__init__(self, r_cut, nlist, name);

        self.group = group;


class respa_lj(respa_pair, lj):
    R""" Respa Lennard-Jones pair potential.
    This class simply extends the hoomd.md.pair.lj class to act as a RespaPotentialPair
    """

    def __init__(self, r_cut, nlist, group, name=None):
        hoomd.util.print_status_line();
        respa_pair.__init__(self, r_cut, nlist, group, name);

        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _multiple_timestep_plugin.RespaPotentialPairLJ(hoomd.context.current.system_definition,
                                                                            self.nlist.cpp_nlist, self.group.cpp_group,
                                                                            self.name);
            self.cpp_class = _multiple_timestep_plugin.RespaPotentialPairLJ;
        else:
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full);
            self.cpp_force = _multiple_timestep_plugin.RespaPotentialPairLJGPU(hoomd.context.current.system_definition,
                                                                               self.nlist.cpp_nlist,
                                                                               self.group.cpp_group, self.name);
            self.cpp_class = _multiple_timestep_plugin.RespaPotentialPairLJGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        self.required_coeff = ['epsilon', 'sigma', 'alpha'];
        self.pair_coeff.set_default_coeff('alpha', 1.0);
