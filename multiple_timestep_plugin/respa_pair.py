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

# TODO:
# * Investigate whether or not you can just override self.cpp_force to be a RespaForceCompute object, or how that even works for the normal pair class.
#      Other wise, we might not be able to inherit from the normal pair class. Yikes.

from hoomd.md.pair import pair

class respa_pair(pair):
    R""" Respa pair potential documentation.

    This class simply extends the hoomd.md.pair.pair class to act as a RespaPotentialPair


    """
    def __init__(self,r_cut,nlist,group,name=None):
        # TODO:
        # * Pass r_cut, nlist, and name to the pair.__init__() funciton.

        self.group