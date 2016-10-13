#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""

Description
-----------

Objects to sample moves from SaltSwap and molecular dynamics. Includes an MCMC sampler class inspired by choderalab/openmmmcmc.
Includes a basic sampler for adjusted mixture sampling for the SaltSwap class.

Notes
-----

The is development.

References
----------

[1] Tan Z., Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics (17 November 2015)

Examples
--------

from openmmtools.testsystems import WaterBox
wbox = WaterBox(box_edge=20,nonbondedMethod=app.PME)
sampler = MCMCSampler(wbox.system,wbox.topology,wbox.positions,delta_chem=710)
sampler.multimove(1000)

Copyright and license
---------------------

@author Gregory A. Ross <gregoryross.uk@gmail.com>

"""

import numpy as np
from saltswap import SaltSwap
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
    def __init__(self, system, topology, positions, temperature = 300*unit.kelvin, pressure = 1*unit.atmospheres, delta_chem = 0, mdsteps = 2000, saltsteps = 0, volsteps = 25,
        ctype = 'CPU', npert = 1, nprop = 0, timestep = 1.0*unit.femtoseconds, propagator = 'GHMC', waterName = "HOH", cationName = 'Na+', anionName = 'Cl-', debug = False):

        self.delta_chem = delta_chem
        self.temperature = temperature
        self.pressure  = pressure
        self.mdsteps = mdsteps
        self.volsteps = volsteps
        self.saltsteps = saltsteps
        self.nprop = nprop

        proplist = ['GHMC','GHMC_old','velocityVerlet']
        if propagator not in proplist:
            raise Exception('NCMC propagator {0} not in supported list {1}'.format(propagator,proplist))

        # Setting the compound integrator:
        if nprop != 0:
            self.integrator = openmm.CompoundIntegrator()
            self.integrator.addIntegrator(openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 2.0*unit.femtoseconds))
            if propagator == 'GHMC':
                self.integrator.addIntegrator(GHMCIntegrator(temperature, 1/unit.picosecond, timestep, nsteps=nprop))
            elif propagator == 'GHMC_old':
                self.integrator.addIntegrator(integrators.GHMCIntegrator(temperature, 1/unit.picosecond, timestep))
            elif propagator=='velocityVerlet':
                self.integrator.addIntegrator(integrators.VelocityVerletIntegrator(timestep*unit.femtoseconds))
            else:
                raise Exception('NCMC propagator {0} not in supported list {1}'.format(propagator,proplist))
            self.integrator.setCurrentIntegrator(0)
        else:
            self.integrator = openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 2.0*unit.femtoseconds)

        # Setting the barostat:
        if pressure is not None:
            self.barostat = openmm.MonteCarloBarostat(pressure, temperature, volsteps)
            system.addForce(self.barostat)

        # Creating the context:
        if ctype == 'CUDA':
            platform = openmm.Platform.getPlatformByName(ctype)
            properties = {'CudaPrecision': 'mixed'}
            self.context = openmm.Context(system, self.integrator, platform, properties)
        else:
            platform = openmm.Platform.getPlatformByName('CPU')
            self.context = openmm.Context(system, self.integrator, platform)
        self.context.setPositions(positions)
        self.context.setVelocitiesToTemperature(temperature)

        # Initialising the saltswap object
        self.saltswap = SaltSwap(system=system,topology=topology,temperature=temperature, delta_chem=delta_chem,integrator=self.integrator,pressure=pressure,
                                 npert=npert, nprop=nprop, propagator = propagator, waterName=waterName, cationName=cationName, anionName=anionName, debug=debug)

    def gen_config(self,mdsteps=None):
        """
        Generate a configuration via one of OpenMM's integrators, and volume move with a frequency as specified in self.barostat.

        Parameters
        ----------
        mdsteps : int
            The number of MD steps to take

        """
        if self.nprop !=0:
            self.integrator.setCurrentIntegrator(0)
        if mdsteps == None:
            self.integrator.step(self.mdsteps)
        else:
            self.integrator.step(mdsteps)


    def gen_label(self,saltsteps=None,delta_chem=None):
        """
        Generate a new number of salt molecules via SaltSwap's code

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

        if self.nprop !=0:
            self.integrator.setCurrentIntegrator(1)
        if saltsteps == None:
            self.saltswap.update(self.context,nattempts=self.saltsteps,cost=cost)
        else:
            self.saltswap.update(self.context,nattempts=saltsteps,cost=cost)

    def move(self,mdsteps=None,saltsteps=None,delta_chem=None):
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
        self.gen_label(saltsteps,delta_chem)


    def multimove(self,nmoves=10,mdsteps=None,saltsteps=None,delta_chem=None):
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
            self.move(mdsteps,saltsteps,delta_chem)

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
    def __init__(self,system, topology, positions, temperature=300*unit.kelvin, pressure=1*unit.atmospheres, delta_chem=0, mdsteps=1000, saltsteps=1, volsteps = 25,
        ctype = 'CPU', npert=0, nprop=0, propagator = 'GHMC', niterations=1000, burnin=100,b=0.7, saltmax = 50):

        super(SaltSAMS, self).__init__(system=system, topology=topology, positions=positions, temperature=temperature, pressure=pressure, delta_chem=delta_chem, mdsteps=mdsteps, saltsteps=saltsteps, volsteps = volsteps,
        ctype = ctype, npert=npert, nprop=nprop,propagator = propagator)

        self.burnin = burnin
        self.b = b
        self.niterations = niterations
        self.step = 1
        self.saltmax = saltmax

        self.zeta = np.zeros(saltmax+1)
        self.pi = np.ones(saltmax+1)/(saltmax+1)

        # Keeping track of the state visited and the values of the vector of zetas
        self.zetatime = [self.zeta]
        self.statetime = []

        self.update_state()

    def update_state(self):
        """
        The find which distribution the Sampler is in, equal to the number of salt pairs. The number of salt pairs
        serves as the index for target density and free energy.
        """
        (junk1,nsalt,junk2) = self.saltswap.getIdentityCounts()
        self.nsalt = nsalt
        self.statetime.append(nsalt)

    def gen_samslabel(self,saltsteps=None):
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
                penalty = ['junk',self.zeta[self.nsalt-1]-self.zeta[self.nsalt]]
            elif self.nsalt == 0:
                penalty = [self.zeta[self.nsalt+1]- self.zeta[self.nsalt],'junk']
            else:
                penalty = [self.zeta[self.nsalt+1]- self.zeta[self.nsalt],self.zeta[self.nsalt-1]-self.zeta[self.nsalt]]
            self.saltswap.attempt_identity_swap(self.context,penalty,self.saltmax)
            self.update_state()

    def adapt_zeta(self):
        """
        Update the free energy estimate for the current state based SAMS binary procedure (equation 9)

        """

        # Burn-in procedure as suggested in equation 15
        if self.step <= self.burnin:
            gain = min(self.pi[self.nsalt],self.step**(-self.b))
        else:
            gain = min(self.pi[self.nsalt],1.0/(self.step - self.burnin + self.burnin**self.b))

        # Equations 4 and 9
        zeta_half = np.array(self.zeta)                        # allows operations to be performed on zeta_half that don't act on zeta
        zeta_half[self.nsalt]  = self.zeta[self.nsalt] + gain/(self.pi[self.nsalt])
        self.zeta = zeta_half - zeta_half[0]

        self.zetatime.append(self.zeta)

    def calibration(self,niterations=None,mdsteps=None,saltsteps=None):
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
        if mdsteps==None: mdsteps = self.mdsteps
        if saltsteps==None: saltsteps = self.saltsteps

        for i in range(niterations):
            self.gen_config(mdsteps)
            self.gen_samslabel(saltsteps)
            self.adapt_zeta()
            self.step += 1