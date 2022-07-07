# Note: This file might be depricated....

print("Importing RESPA plugin. (1/27/2022 11:52 MDT)")

import itertools

from hoomd.respa_plugin import _respa_plugin

from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyFrom, OnlyTypes
from hoomd.integrate import BaseIntegrator
from hoomd.data import syncedlist
from hoomd.md.methods import Method
from hoomd.md.force import Force
from hoomd.md.constrain import Constraint, Rigid


def _preprocess_aniso(value):
    if value is True:
        return "true"
    elif value is False:
        return "false"
    else:
        return value


def _set_synced_list(old_list, new_list):
    old_list.clear()
    old_list.extend(new_list)


class Integrator_Respa(_md._DynamicIntegrator):
    """Enables the RESPA integrator using NVE.
    Args:
        dt (float): Integrator time step size (in time units).
        forces (Sequence[hoomd.md.force.Force, int]): Sequence of
            force/frequency pairs to be applied to the particles in the system.
            The default value of ``None`` initializes an empty list.
        aniso (str or bool): Whether to integrate rotational degrees of freedom
            (bool), default 'auto' (autodetect if there is anisotropic factor
            from any defined active or constraint forces).
        constraints (Sequence[hoomd.md.constrain.Constraint]): Sequence of
            constraint forces applied to the particles in the system.
            The default value of ``None`` initializes an empty list. Rigid body
            objects (i.e. `hoomd.md.constrain.Rigid`) are not allowed in the
            list.
        rigid (hoomd.md.constrain.Rigid): A rigid bodies object defining the
            rigid bodies in the simulation.
    The classes of following modules can be used as elements in `forces`
    - `hoomd.md.angle`
    - `hoomd.md.bond`
    - `hoomd.md.charge`
    - `hoomd.md.dihedral`
    - `hoomd.md.external`
    - `hoomd.md.force`
    - `hoomd.md.improper`
    - `hoomd.md.pair`
    - `hoomd.md.wall`
    - `hoomd.md.special_pair`
    The classes of the following module can be used as elements in `constraints`
    - `hoomd.md.constrain`
    Examples::
        nlist = hoomd.md.nlist.Cell()
        lj1 = hoomd.md.pair.LJ(nlist=nlist)
        lj1.params.default = dict(epsilon=1.0, sigma=1.0)
        lj1.r_cut[('A', 'A')] = 2**(1/6)
        lj2 = hoomd.md.pair.LJ(nlist=nlist)
        lj2.params.default = dict(epsilon=5.0, sigma=3.0)
        lj2.r_cut[('A', 'A')] = 5**(1/6)
        integrator = hoomd.md.Integrator(dt=0.001, forces=[[lj1,1],[lj2,4]])
        sim.operations.integrator = integrator
    Attributes:
        dt (float): Integrator time step size (in time units).
        forces (List[hoomd.md.force.Force]): List of forces applied to the
        particles in the system.
        freqs (List[int]): List of frequencies associated with each of the
        forces given in 'forces'.
        aniso (str): Whether rotational degrees of freedom are integrated.
        constraints (List[hoomd.md.constrain.Constraint]): List of
            constraint forces applied to the particles in the system.
        rigid (hoomd.md.constrain.Rigid): The rigid body definition for the
            simulation associated with the integrator.
    """

    def __init__(self,
                 dt,
                 aniso='auto',
                 forces=None,
                 constraints=None,
                 methods=None,
                 rigid=None):

        if forces == None:
            forceObjects = None
            freqObjects = None
        else:
            forceObjects = []
            freqObjects = []
            for pair in forces:
                forceObjects.append(pair[0])
                freqObjects.append(pair[1])

        super().__init__(forceObjects, constraints, methods, rigid)

        def isInt(obj):
            return isinstance(obj, int)

        self._forceFreqs = syncedlist.SyncedList(
            isInt, syncedlist._PartialGetAttr('_cpp_obj'), iterable=freqObjects)

        self._param_dict.update(
            ParameterDict(dt=float(dt),
                          aniso=OnlyFrom(['true', 'false', 'auto'],
                                         preprocess=_preprocess_aniso),
                          _defaults={"aniso": "auto"}))
        if aniso is not None:
            self.aniso = aniso

    def _attach(self):
        # initialize the reflected c++ class
        self._cpp_obj = _md.RespaIntegrator(
            self._simulation.state._cpp_sys_def, self.dt)

        if self._forces == None:
            raise Exception("There are no forces entered!")
        if self._forceFreqs == None:
            raise Exception("There are no force frequencies entered!")

        if len(self._forceFreqs) != len(self._forces):
            raise IndexError("The number of forces entered (" + str(len(self._forces)) + ") does not match the number of frequencies entered (" + str(len(self._forceFreqs)) + ").")

        for i in range(len(self._forces)):
            force = self._forces[i]
            freq = self._forceFreqs[i]

            self._cpp_obj.addForce(force,freq)

        # Call attach from DynamicIntegrator which attaches forces,
        # constraint_forces, and methods, and calls super()._attach() itself.
        super()._attach()
