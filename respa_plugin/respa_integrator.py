# coding: utf-8

# Maintainer: NathanDavisBarrett

R""" RESPA Integrator

To quote hoomd/integrate.py:

    To integrate the system forward in time, an integration mode must be set. Only one integration mode can be active at
    a time, and the last ``integrate.mode_*`` command before the :py:func:`hoomd.run()` command is the one that will take effect. It is
    possible to set one mode, run for a certain number of steps and then switch to another mode before the next run
    command.

However, the standard integrator 'mode_standard' requires that all forces are computed at each timestep. This is not
compatible with the RESPA algorithm that enables multiple timesteps to be used on different force calculations.

Therefore, the 'mode_respa' will replace the 'mode_standard' class for the purposes of this plugin.

See 'mode_respa' for more details.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.respa_plugin import _respa_plugin
import hoomd;
from hoomd.integrate import _integrator
import copy;
import sys;
import warnings;

class mode_respa(_integrator):
    R""" Enables the RESPA integration method.

    Args:
        dt (float): The time interval of the outter-most force (see description bellow)
        force/relative frequency pairs ([[force, int],...]): A list of force/frequency combinations (see description bellow)
        aniso (bool): Whether to integrate rotational degrees of freedom (bool), default None (autodetect).

    In the RESPA algorithm, forces (and their accompanying particle groups) are associated with a certain timestep
    after which they will be recalculated.

    Due to the nature of the RESPA algorithm various timesteps must be nested with each other like this:

        F1 calculated
        |  F2 calculated
        |  |  F3 calculated
        |  |  |
        |  |  Positions Updated
        |  |  |
        |  |  F3 calculated
        |  |  |
        |  |  Positions Updated
        |  |  |
        |  F2 calculated
        |  |  F3 calculated
        |  |  |
        |  |  Positions Updated
        |  |  |
        |  |  F3 calculated
        |  |  |
        |  |  Positions Updated
        |  |  |
        |  F2 calculated
        |  |  F3 calculated
        |  |  |
        |  |  Positions Updated
        |  |  |
        |  |  F3 calculated
        |  |  |
        |  |  Positions Updated
        |  |  |
        F1 calculated
        |  F2 calculated
        |  |  F3 calculated
        ...

    It follows that the time intervals for each inner force calculation must be a factor of the time interval of the
    outer force. When specifying a force, you must also include the relative frequency of each successive force.

    Take the diagram above for example with the timestep (i.e. dt, the time interval of the outermost force) set
    to 300 seconds. The forces would need to be specified as follows:

        [[F1,1],[F2,3],[F3,6]] (Note that for good practice, the outermost force should have a relative frequency of 1)
                               (Note also that these need not be specified in order. But, once sorted, they must follow
                                the same nested factoring pattern)

    For every F1 calculation, F2 is calculated 3 times. In other words, F2 is calculated 3 times as frequently as F1.
    For every F1 calculation, F3 is calculated 6 times. In other words, F3 is calculated 6 times as frequently as F1.

    1 is a factor of 3
    3 is a factor of 6
    and so on.

    At the moment, only NVE simulations can be done using the mode_respa integrator.
    """

    def __init__(self,dt,forceFreqPairs,aniso=None):
        hoomd.util.print_status_line();

        # initialize base class
        _integrator.__init__(self);

        # Store metadata
        self.dt = dt
        self.forceFreqPairs = forceFreqPairs
        self.aniso = aniso
        self.metadata_fields = ['dt', 'forceFreqPairs', 'aniso']

        # initialize the reflected c++ class
        self.cpp_integrator = _respa_plugin.RespaIntegrator(hoomd.context.current.system_definition, dt);
        self.supports_methods = False;

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        hoomd.util.quiet_status();
        if aniso is not None:
            self.set_params(aniso=aniso)
        hoomd.util.unquiet_status();

        for forceFreqPair in self.forceFreqPairs:
            self.add_force(forceFreqPair)

    ## \internal
    #  \brief Cached set of anisotropic mode enums for ease of access

    _aniso_modes = {
        None: _md.IntegratorAnisotropicMode.Automatic,
        True: _md.IntegratorAnisotropicMode.Anisotropic,
        False: _md.IntegratorAnisotropicMode.Isotropic
    }

    def set_params(self, dt=None, aniso=None):
        R""" Changes parameters of an existing integration mode.
        :param dt (float): New outermost timestep.
        :param aniso (bool): Anisotropic integration mode, default to None (autodetect).
        """

        hoomd.util.print_status_line();
        self.check_initialization();

        # change the parameters
        if dt is not None:
            self.dt = dt
            self.cpp_integrator.setDeltaT(dt);

        if aniso is not None:
            if aniso in self._aniso_modes:
                anisoMode = self._aniso_modes[aniso]
            else:
                hoomd.context.msg.error("integrate.mode_standard: unknown anisotropic mode {}.\n".format(aniso));
                raise RuntimeError("Error setting anisotropic integration mode.");
            self.aniso = aniso
            self.cpp_integrator.setAnisotropicMode(anisoMode)

    def add_force(self,forceFreqPair):
        R""" Adds a force/frequency pair to the reflected c++ class
        :param forceFreqPair ([hoomd Force Object,int]): A force/freq pair. See the documentation for mode_respa for
                                                         for more details.
        """

        force, freq = forceFreqPair
        cpp_force = force.cpp_force

        if force.enabled:
            self.cpp_integrator.addForce(cpp_force, freq)

    def update_forces(self):
        self.check_initialization();

        for f in hoomd.context.current.forces:
            if f.cpp_force is None:
                hoomd.context.msg.error('Bug in hoomd.integrate: cpp_force not set, please report\n');
                raise RuntimeError('Error updating forces');

            if f.log or f.enabled:
                f.update_coeffs();
