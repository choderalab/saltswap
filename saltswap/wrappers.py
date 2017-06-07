#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""

Description
-----------
A set of tools to simplify the application constant salt-concentration molecular dynamics simulations.

Authors
-------
Gregory A. Ross
"""

import numpy as np
from saltswap.swapper import Swapper
import simtk.openmm as openmm
import simtk.unit as unit
from openmmtools import integrators as open_integrators
from scipy import optimize
from saltswap.integrators import GHMCIntegrator


# The parameters required to determine the chemical potential from concentration. These parameters have been calculated
# from self-adjusted mixture sampling simulations using PME with a 10 Angstrom cutoff, Ewald tolerance of 1E-4, and a
# switch width of 1.5 Angstroms.
default_tip4pew_weights = {'fn':np.array([ 0., -318.02312532, -634.90809654, -950.74578027, -1266.27103839,
                                           -1581.38318526, -1896.34568864, -2210.77776154, -2525.0550286,
                                           -2839.12924175, -3153.02481357, -3466.68052032, -3780.49884212,
                                           -4094.0576715, -4407.4606291 , -4720.89693865, -5034.03848188,
                                           -5346.98186477, -5660.10721546, -5973.00551306, -6285.65777448]),
                           'volume':np.array([26.65267978, 26.57112124, 26.50371498, 26.41689722, 26.35040751,
                                             26.28338648, 26.20945854, 26.1474357 , 26.07480588, 26.00300707,
                                             25.93514985, 25.87364533, 25.81137662, 25.74550946, 25.68789879,
                                             25.61304901, 25.55691636, 25.49593746, 25.42924935, 25.37536792,
                                              25.3079591])}
default_tip3p_weights = {'fn':np.array([0., -320.03292132, -638.69174344, -956.55600114, -1273.88697038, -1590.77832955,
                                        -1907.46929657, -2223.79014252, -2539.73095265, -2855.51349456, -3171.15183542,
                                        -3486.63899421, -3801.8856171 , -4117.04030733, -4432.24835394, -4747.33615489,
                                        -5062.26928236, -5377.151068  , -5691.73414011, -6006.27264767,
                                        -6320.76906322]),
                         'volume':np.array([26.95692854, 26.88854455, 26.81861493, 26.75631525, 26.68304869,
                                            26.62441529, 26.55923851, 26.50940181, 26.44217547, 26.38498465, 26.3304819,
                                            26.2684076, 26.20687082, 26.15909901, 26.1107472 , 26.06642252, 26.00498962,
                                            25.93755518, 25.8777354 , 25.81917845, 25.75393419])}

class Salinator(object):
    """
    A user-friendly wrapper for performing constant-salt-concentration simulations using the saltswap machinery.
    Use this object to neutralize a system with the saltswap-specific ion topologies, initialize the ion concentration,
    and perform the NCMC accelerated insertiona and deletion of salt.

    Example
    -------
    A constant-salt concentration simulation on the tip3p DHFR test system.
    >>> from simtk import openmm, unit
    >>> from openmmtools import testsystems
    >>> from openmmtools import integrators

    Set the thermondynamic parameters
    >>> temperature = 300.0 * unit.kelvin
    >>> pressure = 1.0 * unit.atmospheres
    >>> salt_concentration = 0.2 * unit.molar

    Extract the test system:
    >>> testobj = getattr(testsystems, 'DHFRExplicit')
    >>> testsys = testobj(nonbondedMethod=openmm.app.PME, cutoff=10 * unit.angstrom, ewaldErrorTolerance=1E-4,
    >>> ...               switch_width=1.5 * unit.angstrom)
    >>> testsys.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    Create the compound integrator to perform molecular dynamics and the NCMC propagation. The choice of the Langevin
    splitting is important here, as 'V R O R V' ensures low configuration space error for the unmetropolized dynamics.
    >>> langevin = integrators.LangevinIntegrator(splitting='V R O R V', temperature=temperature,
    >>> ...                                       measure_shadow_work=False, measure_heat=False)
    >>> ncmc_langevin = integrators.ExternalPerturbationLangevinIntegrator(splitting='V R O R V',
    >>> ...                              temperature=temperature, measure_shadow_work=False, measure_heat=False)
    >>> integrator = openmm.CompoundIntegrator()
    >>> integrator.addIntegrator(langevin)
    >>> integrator.addIntegrator(ncmc_langevin)

    Create the context:
    >>> context = openmm.Context(testsys.system, integrator)
    >>> context.setPositions(testsys.positions)
    >>> context.setVelocitiesToTemperature(temperature)

    Create the salinator:
    >>> salinator = Salinator(context=context, system=testsys.system, topology=testsys.topology,
    >>> ...                            ncmc_integrator=ncmc_langevin, salt_concentration=salt_concentration,
    >>> ...                            pressure=pressure, temperature=temperature, water_name='WAT')

    Neutralize the system and initialize the number of salt pairs near to the expected amount.
    >>> salinator.neutralize()
    >>> salinator.initialize_concentration()

    Runing a short simulation by mixing MD and MCMC saltswap moves.
    >>> for i in range(100):
    >>> ...     langevin.step(2000)
    >>> ...     salinator.update(nattempts=1)
    """

    def __init__(self, context, system, topology, ncmc_integrator, salt_concentration, pressure, temperature,
                 npert=10000, water_name="HOH", cation_name='Na+', anion_name='Cl-', calibration_weights=None):

        """
        Parameters
        ----------
        context: simtk.openmm.openmm.Context
            The simulation context
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        topology : simtk.openmm.app.topology
             Topology of the system
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature to be simulated.
        ncmc_integrator : simtk.openmm.integrator
            The integrator used for NCMC propagation of insertion and deletions
        salt_concentration: simtk.unit.quantity.Quantity
            The macroscopic salt concentration that the simulation will be coupled to.
        pressure : simtk.unit.Quantity compatible with atmospheres, optional, default=None
            For explicit solvent simulations, the pressure.
        npert : int
            Number of ncmc perturbation kernels. Set to 1 for instantaneous switching
        water_name = str, optional, default='HOH'
            Name of water residue that will be exchanged with salt
        cation_name : str, optional, default='Na+'
            Name of cation residue from which parameters are to be taken.
        anion_name : str, optional, default='Cl-'
            Name of anion residue from which parameters are to be taken.
        calibration_weights: dict
            Dictionary containing the salt insertion free energies and volumes from the calibration of the chemical
            potential.
        """
        # OpenMM system objects
        self.context = context
        self.system = system

        # MCMC and NCMC parameters
        self.npert = npert
        # NCMC will not be performed for 1 perturbation.
        if self.npert > 1:
            # Number of propagation steps per perturbation.
            # nprop > 1 only suported for saltswap.integrators.GHMCIntegrator.
            nprop = 1
        elif self.npert == 1:
            # No propagation steps per perturbation, which mean no NCMC, ansd instantaneous insertion/deletion attempts.
            nprop = 0
        else:
            raise Exception('Invalid number of perturbation steps specified. Must be at least 1.')

        # Initialize the driver for exchanging salt and water. Setting delta_chem to 0.0 for now. Updated below:
        self.swapper = Swapper(system=self.system, topology=topology, temperature=temperature,
                               delta_chem=0.0, ncmc_integrator=ncmc_integrator, pressure=pressure,
                               nattempts_per_update=1, npert=self.npert, nprop=nprop, work_measurement='internal',
                               waterName=water_name, cationName=cation_name, anionName=anion_name)

        # Constant salt parameters
        self.salt_concentration = salt_concentration
        self.chemical_potential = self._get_chemical_potential(calibration_weights)
        self.swapper.delta_chem = self.chemical_potential

    def _get_chemical_potential(self, calibration_weights):
        """
        Extract the required chemical potential from the specified macroscopic salt concentration.
        """
        if calibration_weights is not None:
            volume = calibration_weights['volume']
            fn = calibration_weights['fn']
        else:
            # Guess the chemical potential of water
            # TODO: raise warning that certain non-bonded interaction parameters are expected?
            water_params = self.swapper.water_parameters
            if len(water_params) == 3:
                if water_params[0]['charge'] + 0.834 == 0.0:
                    # Assume TIP3P at default simulation parameters
                    volume = default_tip3p_weights['volume']
                    fn = default_tip3p_weights['fn']
                else:
                    raise Exception('Default parameters not available for three-site water model.')
            elif len(water_params) == 4:
                if water_params[3]['charge'] + 1.04844 == 0.0:
                    # Assume TIP4Pew at default simulation parameters
                    volume = default_tip4pew_weights['volume']
                    fn = default_tip4pew_weights['fn']
                else:
                    raise Exception('Default parameters not available for four-site water model.')

        delta_chem = self.invert_concentration(self.salt_concentration, fn, volume)

        return delta_chem

    @staticmethod
    def predict_concentration(delta_chem, fn, volume):
        """
        Calculate the average concentration of salt (in M) at a given chemical potential.

        Parameters
        ----------
        delta_chem: float
            The difference between the chemical potential of two water molecules and anion and cation (in kT).
        fn: numpy.ndarray
            The free energy exchange salt and two water molecules (in kT).
        volume: numpu.ndarray
            The mean volume (in nm**3) as a function of the number of salt pairs.

        Returns
        -------
        concentration: float
            The mean concentration (in mols/litre) of salt that occurs with the supplied parameters.

        """
        nsalt = np.arange(0, len(fn))
        exponents = -delta_chem * nsalt - fn
        a = np.max(exponents)
        numerator = np.sum(nsalt * np.exp(exponents - a))
        denominator = np.sum(volume * np.exp(exponents - a))
        concentration = numerator/denominator * 1.66054

        return float(concentration)

    @staticmethod
    def invert_concentration(concentration, fn, volume, initial_guess=318):
        """
        Extract the chemical potential required to achieve the specified macroscopic salt concentration. Numerically
        inverts the predict_concentration function.

        Parameters
        ----------
        concentration: simtk.unit.quantity.Quantity or float
            The desired macroscopic salt concentration. If float, assume units are in mols/litre.
        fn: numpy.ndarray
            The free energy exchange salt and two water molecules (in kT).
        volume: numpu.ndarray
            The mean volume (in nm**3) as a function of the number of salt pairs.
        initial_guess: float
            The initial guess of the required chemical potential

        Returns
        -------
        delta_chem: float
            The chemical potential required to achieve the desierd macroscopic salt concentration in saltswap\
        """
        if type(concentration) == unit.quantity.Quantity:
            c = concentration.in_units_of(unit.molar)._value
        elif type(concentration) == float or type(concentration) == int:
            # Assume units are in molar (mols/litre)
            c = float(concentration)
        else:
            raise Exception('Cannot recognize the type of concentration: "{0}"'.format(type(concentration)))

        def loss(mu):
            return (c - Salinator.predict_concentration(mu, fn, volume)) ** 2

        delta_chem = optimize.minimize(loss, initial_guess, method='BFGS', options={'gtol': 1e-07}).x[0]

        return float(delta_chem)

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

    def _add_ion(self, ion_type, water_index, nonbonded_force):
        """
        Swap a water molecule, specified by its index, for either a cation or anion.

        Parameters
        ----------
        ion_type: str
        water_index: int
        nonbonded_force: simtk.openmm.openmm.NonbondedForce
        """
        stage = self.npert

        if ion_type == 'cation':
            parameter_path = self.swapper.wat2cat_parampath
            self.swapper.stateVector[water_index] = 1        # Update state vector, which indexes cations with 1.
        elif ion_type == 'anion':
            parameter_path = self.swapper.wat2an_parampath
            self.swapper.stateVector[water_index] = 2        # Update state vector, which indexes anions with 2

        # Get the residue that will be turned into an ion
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
                    self._add_ion('cation', water_index, nonbonded_force)
            else:
                for water_index in water_indices:
                    self._add_ion('anion', water_index, nonbonded_force)

    def initialize_concentration(self):
        """
        Instantaneously insert salt pairs to approximately match the input concentration. This is will be the starting
        number for the MCMC salt insertion and deletion. These ions are added on top of the neutralizing ions.
        """

        # Estimate how many salt should be added TODO: use the actual concentration for the appropriate model?
        water_conc = 55.4 # Approximate concentration of water
        nwaters = np.sum(self.swapper.stateVector == 0)
        nsalt = int(np.floor(nwaters * self.salt_concentration / (water_conc * unit.molar)))

        # Select which water molecules will be converted to anions anc cations
        # TODO: Can I ensure salt isn't added inside the protein?
        water_indices = np.random.choice(a=np.where(self.swapper.stateVector == 0)[0], size=2*nsalt, replace=False)
        cation_indices = water_indices[0:nsalt]
        anion_indices = water_indices[nsalt:]

        # Insert the salt!
        nonbonded_force = self._get_nonbonded_force()
        for a_ind, c_ind in zip(cation_indices, anion_indices):
            self._add_ion('cation', c_ind, nonbonded_force)
            self._add_ion('anion', a_ind, nonbonded_force)

    def update(self, nattempts=None, chemical_potential=None, saltmax=None):
        """
        Perform MCMC salt insertion/deletion moves.
        """
        self.swapper.update(self.context, nattempts=nattempts, cost=chemical_potential, saltmax=saltmax)


class MCMCSampler(object):
    """
    Simplified wrapper for mixing molecular dynamics and MCMC saltswap moves. Greatly simplifies the setup procedure,
    at the expense of flexibility and generality.

    Example
    --------
    Mix MD and MCMC saltswap moves on a box of water.
    >>> from openmmtools.testsystems import WaterBox
    >>> wbox = WaterBox(box_edge=20,nonbondedMethod=app.PME)
    >>> sampler = MCMCSampler(wbox.system,wbox.topology,wbox.positions,delta_chem=710)
    >>> sampler.multimove(1000)

    """
    def __init__(self, system, topology, positions, temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres,
                 delta_chem=0, mdsteps=2000, saltsteps=0, volsteps=25, saltmax=None, platform='CPU', npert=1, nprop=0,
                 timestep=1.5 * unit.femtoseconds, propagator='Langevin', waterName='HOH', cationName='Na+',
                 anionName='Cl-'):
        """
        Initialize the MCMC sampler for MD and saltswap MCMC moves. Context creation is handled under the hood.

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
    DEPRICATED
    TODO: remove and replace with machinery from the BAMS repo.

    Implementation of self-adjusted mixture sampling for exchanging water and salts in a grand canonical methodology. The
    mixture is over integer increments of the number of salt molecules up to a specified maximum. The targed density is
    currently hard coded in as uniform over the number of salt molecules.

    References
    ----------
    [1] Z. Tan, Optimally adjusted mixture sampling and locally weighted histogram analysis
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
