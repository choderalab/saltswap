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
from saltswap.swapper import Swapper
import simtk.openmm as openmm
import simtk.unit as unit
from openmmtools import integrators as open_integrators

from saltswap.integrators import GHMCIntegrator


class Salinator(object):
    """
    A usuable wrapper for performing constant-salt-concentration simulations
    """

    def __init__(self, context, system, topology, ncmc_integrator, salt_concentration, pressure, temperature, npert,
                 water_name="HOH", cation_name='Na+', anion_name='Cl-'):

        """
        Parameters
        ----------
        context: simtk.openmm.openmm.Context
        system: simtk.openmm.System
        topology: simtk.openmm.app.topology
        ncmc_integrator: simtk.openmm.integrator
        salt_concentration: simtk.unit
        pressure: simtk.unit
        temperature: simtk.unit
        npert: int
        water_name: str
        cation_name: str
        anion_name: str
        """
        # TODO: read in water model?
        # OpenMM system objects
        self.context = context
        self.system = system

        # Thermodynamic constraint
        self.salt_concentration = salt_concentration

        # MCMC and NCMC parameters
        self.npert = npert

        # Initialize the driver for exchanging salt and water
        chemical_potential = self._get_chemical_potential()
        self.swapper = Swapper(system=self.system, topology=topology, temperature=temperature,
                               delta_chem=chemical_potential, ncmc_integrator=ncmc_integrator, pressure=pressure,
                               nattempts_per_update=1, npert=self.npert, nprop=0,  work_measurement='internal',
                               waterName=water_name, cationName=cation_name, anionName=anion_name)

    def _get_chemical_potential(self):
        """
        Extract the required chemical potential from the specified macroscopic salt concentration.
        """
        #TODO: this functionality needs to be added.
        return 0.0

    def _get_nonbonded_force(self):
        """
        Extract the OpenMM non-bonded force from the system.

        Returns
        -------
        nonbonded_force: simtk.openmm.openmm.NonbondedForce
            the non-bonded force
        """
        nonbonded_force = None
        for force_index in range(self.system.getNumForces()):
            force = self.system.getForce(force_index)
            if force.__class__.__name__ == 'NonbondedForce':
                nonbonded_force = force

        if nonbonded_force is None:
            raise Exception('System does not contain a non-bonded force.')

        return nonbonded_force

    def _get_system_charge(self, nonbonded_force):
        """
        Calculate the total charge of the system.

        Parameters
        ----------
        nonbonded_force:

        Returns
        -------
        total_charge: int
            the total charge of the system.
        """
        total_charge = 0.0
        for i in range(self.system.getNumParticles()):
            total_charge += nonbonded_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        return int(np.floor(0.5 + total_charge))

    def _add_cation(self, water_index, nonbonded_force):
        """
        Insert a cation and update the system.
        :param water_index:
        :param nonbonded_force:
        :return:
        """
        stage = self.npert
        parameter_path = self.swapper.wat2cat_parampath
        molecule = [atom for atom in self.swapper.mutable_residues[water_index].atoms()]

        # Change the water parameters to be that of the cations
        atom_index = 0
        for atom in molecule:
            nonbonded_force.setParticleParameters(atom.index,
                                                        charge=parameter_path[atom_index]['charge'][stage],
                                                        sigma=parameter_path[atom_index]['sigma'][stage],
                                                        epsilon=parameter_path[atom_index]['epsilon'][stage])
            atom_index += 1

        # Push these new parameters to the context
        nonbonded_force.updateParametersInContext(self.context)

        # Update state vector, which indexes cations with 1.
        self.swapper.stateVector[water_index] = 1

    def _add_anion(self, water_index, nonbonded_force):
        """
        Insert an anion and update the system.
        :param water_index:
        :param nonbonded_force:
        :return:
        """
        stage = self.npert
        parameter_path = self.swapper.wat2an_parampath
        molecule = [atom for atom in self.swapper.mutable_residues[water_index].atoms()]

        # Change the water parameters to be that of the cations
        atom_index = 0
        for atom in molecule:
            nonbonded_force.setParticleParameters(atom.index, charge=parameter_path[atom_index]['charge'][stage],
                                                              sigma=parameter_path[atom_index]['sigma'][stage],
                                                              epsilon=parameter_path[atom_index]['epsilon'][stage])
            atom_index += 1

        # Push these new parameters to the context
        nonbonded_force.updateParametersInContext(self.context)

        # Update state vector, which indexes anions with 2.
        self.swapper.stateVector[water_index] = 2

    def neutralize(self):
        """
        Neutralize the system.
        """
        # Get the non-bonded force:
        nonbonded_force = self._get_nonbonded_force()
        # Get the total charge of the system
        total_charge = self._get_system_charge(nonbonded_force)

        if total_charge != 0:
            # Choose which water molecules will be swapped for neutralizing counter-ions
            water_indices = np.random.choice(a=np.where(self.swapper.stateVector == 0)[0], size=abs(total_charge), replace=False)

            # Change the selected waters into Na+ or Cl- depending on the total charge.
            if total_charge < 0:
                for water_index in water_indices:
                    self._add_cation(water_index, nonbonded_force)
            else:
                for water_index in water_indices:
                    self._add_anion(water_index, nonbonded_force)

    def insert_to_concentration(self):
        """
        Instantaneously insert salt pairs to approximately match the input concentration. This is will be the starting
        number for the MCMC salt insertion and deletion. These ions are added on top of the neutralizing ions.
        """

        # Estimate how many salt should be added TODO: use the actual concentration for the appropriate model?
        water_conc = 55.4 # Approximate concentration of the water model
        nwaters = np.sum(self.swapper.stateVector == 0)
        nsalt = int(np.floor(nwaters * self.salt_concentration / (water_conc * unit.molar)))

        # Select which water molecules will be converted TODO: Can I ensure salt isn't added inside the protein?
        water_indices = np.random.choice(a=np.where(self.swapper.stateVector == 0)[0], size=nsalt, replace=False)

        # Insert the salt!
        nonbonded_force = self._get_nonbonded_force()
        for water_index in water_indices:
            self._add_cation(water_index, nonbonded_force)
            self._add_cation(water_index, nonbonded_force)

    def update(self):
        """
        Perform an MCMC salt insertion/deletion move.
        :return:
        """
        pass

class MCMCSampler(object):
    """
    Basics for molecular dynamics and saltswap swapper moves.
    """
    def __init__(self, system, topology, positions, temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres,
                 delta_chem=0, mdsteps=2000, saltsteps=0, volsteps=25, saltmax=None, platform='CPU', npert=1, nprop=0,
                 timestep=1.5 * unit.femtoseconds, propagator='Langevin', waterName='HOH', cationName='Na+',
                 anionName='Cl-'):
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
            The difference in chemical potential for swapping 2 water molecules for Na Cl. If it is a float, it is
            assumed to be in thermal.
        mdsteps: int
            The number steps of molecular dynamics to take during one iteration
        saltsteps: int
            The number of insertion/deletion attempts to make during one iteration
        volsteps: int
            The frequency of volume moves during a series of MD steps
        saltmax: int
            The maximum number of salt ion pairs that can be inserted into the system. If None, then the maximum number
            is approximately half the number of water molecules.
        npert : integer
            Number of ncmc perturbation kernels. Set to 1 for instantaneous switching
        nprop : integer
            Number of propagation kernels (MD steps) per ncmc perturbation kernel. Set to 0 for instantaneous switching
        timestep : simtk.unit.Quantity with units compatible with femtoseconds
            Timestep to use for ncmc switching
        propagator : str
            The name of the ncmc propagator
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
        self.saltmax = saltmax
        collision_rate = 1 / unit.picosecond


        # Only supporting two two types of integrators.
        proplist = ['GHMC', 'Langevin']
        if propagator not in proplist:
            raise Exception('ncmc propagator {0} not in supported list {1}'.format(propagator, proplist))

        platform_types = ['CUDA', 'OpenCL', 'CPU']
        if platform not in platform_types:
            raise Exception(
                'platform type "{0}" not recognized. Re-enter --platform with a selection from {1}.'.format(platform,
                                                                                                            platform_types))

        # Setting the compound integrator:
        if nprop != 0:
            # NCMC will be used to insert/delete salt
            self.integrator = openmm.CompoundIntegrator()
            if propagator == proplist[0]:
                self.integrator.addIntegrator(GHMCIntegrator(temperature, collision_rate, timestep, nsteps=1))
                ncmc_integrator = GHMCIntegrator(temperature, collision_rate, timestep, nsteps=nprop)
                self.integrator.addIntegrator(ncmc_integrator)
            elif propagator == proplist[1]:
                self.integrator.addIntegrator(open_integrators.LangevinIntegrator(splitting="V R O R V",
                                                                                  temperature=temperature,
                                                                                  timestep=timestep,
                                                                                  collision_rate=collision_rate))
                ncmc_integrator = open_integrators.ExternalPerturbationLangevinIntegrator(splitting="V R O R V",
                                                                                                      temperature=temperature,
                                                                                                      timestep=timestep,
                                                                                                      collision_rate=collision_rate)
                self.integrator.addIntegrator(ncmc_integrator)
            self.integrator.setCurrentIntegrator(0)
        else:
            ncmc_integrator = None
            if propagator == proplist[0]:
                self.integrator = GHMCIntegrator(temperature, collision_rate, timestep, nsteps=1)
            elif propagator == proplist[1]:
                self.integrator = open_integrators.LangevinIntegrator(splitting="V R O R V",
                                                                                  temperature=temperature,
                                                                                  timestep=timestep,
                                                                                  collision_rate=collision_rate)
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
        self.swapper = Swapper(system=system, topology=topology, temperature=temperature, delta_chem=delta_chem,
                               ncmc_integrator=ncmc_integrator, pressure=pressure,
                               npert=npert, nprop=nprop, work_measurement='internal', waterName=waterName,
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
            self.swapper.update(self.context, nattempts=self.saltsteps, cost=cost, saltmax=self.saltmax)
        else:
            self.swapper.update(self.context, nattempts=saltsteps, cost=cost, saltmax=self.saltmax)

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
        (junk1, nsalt, junk2) = self.swapper.get_identity_counts()
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
            self.swapper.attempt_identity_swap(self.context, penalty, self.saltmax)
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
