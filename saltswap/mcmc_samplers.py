#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""

Description
-----------

Objects to sample moves from Swapper and molecular dynamics. Includes an MCMC sampler class inspired by choderalab/openmmmcmc.
Includes a basic sampler for adjusted mixture sampling for the Swapper class.

Notes
-----

The is development.

References
----------

[1] Tan Z., Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics (17 November 2015)

Example
--------

from openmmtools.testsystems import WaterBox
wbox = WaterBox(box_edge=20,nonbondedMethod=app.PME)
sampler = MCMCSampler(wbox.system,wbox.topology,wbox.positions,delta_chem=710)
sampler.multimove(1000)

Copyright and license
---------------------

@author Gregory A. Ross <gregory.ross@choderalab.org>

"""

import numpy as np
from swapper import Swapper
import simtk.openmm as openmm
import simtk.unit as unit
from openmmtools import integrators

from integrators import GHMCIntegrator


class MCMCSampler(object):
    """
    Wrapper for MD and saltswap moves.

    Attributes
    ----------
    n_adaptations : int
        Number of times the relative free energies have been adapted.
    state_counts : np.array
        Histogram of the expected weights of current states.
    References
    ----------

    """

    def __init__(self, system, topology, positions, temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres,
                 delta_chem=0, mdsteps=2000, saltsteps=0, volsteps=25,
                 platform='CPU', npert=1, nprop=0, ncmc_timestep=1.0 * unit.femtoseconds, propagator='GHMC',
                 waterName="HOH", cationName='Na+', anionName='Cl-'):
        """
        Initialize a Monte Carlo titration driver for semi-grand ensemble simulation.

        Parameters
        ----------
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        topology : simtk.openmm.app.topology
             Topology of the system
        positions : list or numpy.array
            The coordinates of each atom in the system
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature to be simulated.
        integrator : simtk.openmm.integrator
            The integrator used for dynamics outside of Swapper
        pressure : simtk.unit.Quantity compatible with atmospheres, optional, default=None
            For explicit solvent simulations, the pressure.
        delta_chem : float or unit.Quantity
            The difference in chemical potential for swapping 2 water molecules for Na Cl.
            If it is a float, it is assumed to be in units of kT.
        npert : integer
            Number of _ncmc perturbation kernels. Set to 1 for instantaneous switching
        nprop : integer
            Number of propagation kernels (MD steps) per _ncmc perturbation kernel. Set to 0 for instantaneous switching
        ncmc_timestep : simtk.unit.Quantity with units compatible with femtoseconds
            Timestep to use for _ncmc switching
        propagator : str
            The name of the _ncmc propagator
        waterName = str, optional, default='HOH'
            Name of water residue that will be exchanged with salt
        cationName : str, optional, default='Na+'
            Name of cation residue from which parameters are to be taken.
        anionName : str, optional, default='Cl-'
            Name of anion residue from which parameters are to be taken.
        """

        self.delta_chem = delta_chem
        self.temperature = temperature
        self.pressure = pressure
        self.mdsteps = mdsteps
        self.volsteps = volsteps
        self.saltsteps = saltsteps
        self.nprop = nprop

        # Exceptions
        proplist = ['GHMC', 'GHMC_old', 'GHMC_save_work_per_step', 'velocityVerlet']
        if propagator not in proplist:
            raise Exception('_ncmc propagator {0} not in supported list {1}'.format(propagator, proplist))

        platform_types = ['CUDA', 'OpenCL', 'CPU']
        if platform not in platform_types:
            raise Exception(
                'platform type "{0}" not recognized. Re-enter --platform with a selection from {1}.'.format(platform,
                                                                                                            platform_types))

        # Setting the compound integrator:
        if nprop != 0:
            self.integrator = openmm.CompoundIntegrator()
            self.integrator.addIntegrator(
                GHMCIntegrator(temperature, 1 / unit.picosecond, 2.0 * unit.femtoseconds, nsteps=1))
            if propagator == 'GHMC':
                self.integrator.addIntegrator(
                    GHMCIntegrator(temperature, 1 / unit.picosecond, ncmc_timestep, nsteps=nprop))
            elif propagator == 'GHMC_old':
                self.integrator.addIntegrator(
                    integrators.GHMCIntegrator(temperature, 1 / unit.picosecond, ncmc_timestep))
            elif propagator == 'velocityVerlet':
                self.integrator.addIntegrator(integrators.VelocityVerletIntegrator(ncmc_timestep * unit.femtoseconds))
            else:
                raise Exception('_ncmc propagator {0} not in supported list {1}'.format(propagator, proplist))
            self.integrator.setCurrentIntegrator(0)
        else:
            self.integrator = GHMCIntegrator(temperature, 1 / unit.picosecond, 2.0 * unit.femtoseconds, nsteps=1)

        # Setting the barostat:
        if pressure is not None:
            self.barostat = openmm.MonteCarloBarostat(pressure, temperature, volsteps)
            system.addForce(self.barostat)

        # Creating the context:
        if platform == 'CUDA':
            platform = openmm.Platform.getPlatformByName(platform)
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
            properties = {'CudaPrecision': 'mixed'}
            self.context = openmm.Context(system, self.integrator, platform, properties)
        elif platform == 'OpenCL':
            platform = openmm.Platform.getPlatformByName('OpenCL')
            properties = {'OpenCLPrecision': 'mixed'}
            self.context = openmm.Context(system, self.integrator, platform, properties)
        else:
            platform = openmm.Platform.getPlatformByName('CPU')
            self.context = openmm.Context(system, self.integrator, platform)
        self.context.setPositions(positions)
        self.context.setVelocitiesToTemperature(temperature)

        # Initialising the saltswap object
        self.saltswap = Swapper(system=system, topology=topology, temperature=temperature, delta_chem=delta_chem,
                                integrator=self.integrator, pressure=pressure,
                                npert=npert, nprop=nprop, propagator=propagator, waterName=waterName,
                                cationName=cationName, anionName=anionName)

    def gen_config(self, mdsteps=None):
        """
        Generate a configuration via one of OpenMM's integrators, and volume move with a frequency as specified in self.barostat.

        Parameters
        ----------
        mdsteps : int
            The number of MD steps to take

        """
        if self.nprop != 0:
            self.integrator.setCurrentIntegrator(0)
        if mdsteps == None:
            self.integrator.step(self.mdsteps)
        else:
            self.integrator.step(mdsteps)

    def gen_label(self, saltsteps=None, delta_chem=None):
        """
        Generate a new number of salt molecules via Swapper's code

        Parameters
        ----------
        saltsteps : int
            The number of salt insertion/deletion attempts
        delta_chem : float
            The difference in chemical potential between salt and 2 water molecules

        """
        if delta_chem == None:
            cost = self.delta_chem
        else:
            cost = delta_chem

        if self.nprop != 0:
            self.integrator.setCurrentIntegrator(1)
        if saltsteps == None:
            self.saltswap.update(self.context, nattempts=self.saltsteps, cost=cost)
        else:
            self.saltswap.update(self.context, nattempts=saltsteps, cost=cost)

    def move(self, mdsteps=None, saltsteps=None, delta_chem=None):
        """
        Generate a move composed of configuration, volume, and salt insertion/deletion attempts

        Parameters
        ----------
        mdsteps : int
            The number of MD steps to take
        saltsteps : int
            The number of salt insertion/deletion attempts
        delta_chem : float
            The difference in chemical potential between salt and 2 water molecules
        """
        self.gen_config(mdsteps)
        self.gen_label(saltsteps, delta_chem)

    def multimove(self, nmoves=10, mdsteps=None, saltsteps=None, delta_chem=None):
        """
        Generate a many moves over the sampling dimensions.

        Parameters
        ----------
        nmoves : int
            The number of iterations combination moves
        mdsteps : int
            The number of MD steps to take
        saltsteps : int
            The number of salt insertion/deletion attempts
        delta_chem : float
            The difference in chemical potential between salt and 2 water molecules
        """
        for i in range(nmoves):
            self.move(mdsteps, saltsteps, delta_chem)


class SaltSAMS(MCMCSampler):
    """
    Implementation of self-adjusted mixture sampling for exchanging water and salts in a grand canonical methodology. The
    mixture is over integer increments of the number of salt molecules up to a specified maximum. The targed density is
    currently hard coded in as uniform over the number of salt molecules.

    References
    ----------
    .. [1] Z. Tan, Optimally adjusted mixture sampling and locally weighted histogram analysis
        DOI: 10.1080/10618600.2015.111397
    """

    def __init__(self, system, topology, positions, temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres,
                 delta_chem=0, mdsteps=1000, saltsteps=1, volsteps=25,
                 platform='CPU', npert=0, nprop=0, propagator='GHMC', niterations=1000, burnin=100, b=0.7, saltmax=50):

        super(SaltSAMS, self).__init__(system=system, topology=topology, positions=positions, temperature=temperature,
                                       pressure=pressure, delta_chem=delta_chem, mdsteps=mdsteps, saltsteps=saltsteps,
                                       volsteps=volsteps,
                                       platform=platform, npert=npert, nprop=nprop, propagator=propagator)

        self.burnin = burnin
        self.b = b
        self.niterations = niterations
        self.step = 1
        self.saltmax = saltmax

        self.zeta = np.zeros(saltmax + 1)
        self.pi = np.ones(saltmax + 1) / (saltmax + 1)

        # Keeping track of the state visited and the values of the vector of zetas
        self.zetatime = [self.zeta]
        self.statetime = []

        self.update_state()

    def update_state(self):
        """
        The find which distribution the Sampler is in, equal to the number of salt pairs. The number of salt pairs
        serves as the index for target density and free energy.
        """
        (junk1, nsalt, junk2) = self.saltswap.get_identity_counts()
        self.nsalt = nsalt
        self.statetime.append(nsalt)

    def gen_samslabel(self, saltsteps=None):
        """
        Attempt a move to add or remove salt molecules. In labelled mixture sampling parlance, a new label is generated
        using a local jump strategy. This function overwrites the gen_label in the 'Sample' class, so that the free
        energy estimates (zeta) can be used to weight transitions.

        Parameters
        ----------
        saltsteps: int
            The number of water-salt swaps that will be attempted
        """
        for step in range(saltsteps):
            if self.nsalt == self.saltmax:
                penalty = ['junk', self.zeta[self.nsalt - 1] - self.zeta[self.nsalt]]
            elif self.nsalt == 0:
                penalty = [self.zeta[self.nsalt + 1] - self.zeta[self.nsalt], 'junk']
            else:
                penalty = [self.zeta[self.nsalt + 1] - self.zeta[self.nsalt],
                           self.zeta[self.nsalt - 1] - self.zeta[self.nsalt]]
            self.saltswap.attempt_identity_swap(self.context, penalty, self.saltmax)
            self.update_state()

    def adapt_zeta(self):
        """
        Update the free energy estimate for the current state based SAMS binary procedure (equation 9)

        """

        # Burn-in procedure as suggested in equation 15
        if self.step <= self.burnin:
            gain = min(self.pi[self.nsalt], self.step ** (-self.b))
        else:
            gain = min(self.pi[self.nsalt], 1.0 / (self.step - self.burnin + self.burnin ** self.b))

        # Equations 4 and 9
        zeta_half = np.array(self.zeta)  # allows operations to be performed on zeta_half that don't act on zeta
        zeta_half[self.nsalt] = self.zeta[self.nsalt] + gain / (self.pi[self.nsalt])
        self.zeta = zeta_half - zeta_half[0]

        self.zetatime.append(self.zeta)

    def calibration(self, niterations=None, mdsteps=None, saltsteps=None):
        """
        Parameters
        ----------
        niterations: int
            The number total calibration steps, where a step consists of sampling configuration, sampling label, and
            adapting zeta.
        mdsteps: int
            The number of molecular dynamics steps used to generate a new configuration
        saltsteps: int
            The number of salt-water exchanges used when updating the label.
        """

        if niterations == None: niterations = self.niterations
        if mdsteps == None: mdsteps = self.mdsteps
        if saltsteps == None: saltsteps = self.saltsteps

        for i in range(niterations):
            self.gen_config(mdsteps)
            self.gen_samslabel(saltsteps)
            self.adapt_zeta()
            self.step += 1
