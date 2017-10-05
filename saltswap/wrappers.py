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
from saltswap.sams_adapter import SAMSAdaptor

# The parameters required to determine the chemical potential from concentration. These parameters have been calculated
# from self-adjusted mixture sampling simulations using PME with a 10 Angstrom cutoff, Ewald tolerance of 1E-4, and a
# switch width of 1.5 Angstroms.
# The free energies are in units of kT at 300K, and volume is in Angstroms cubed.
default_tip4pew_weights = {'fn': np.array([0., -318.05881688, -634.82257946, -950.86394367, -1266.40212151,
                                          -1581.46300874, -1896.27363003, -2210.82650563, -2525.13639012,
                                          -2839.31919345, -3153.29282184, -3467.08768458, -3780.72288485,
                                          -4094.30848518, -4407.73750209, -4721.06504457, -5034.27217415,
                                          -5347.36971538, -5660.4642781, -5973.41366002, -6286.22365696]),
                           'volume': np.array([26.65326186, 26.57516119, 26.50819146, 26.42635379, 26.35226623,
                                               26.28321964, 26.21192966, 26.14517528, 26.07064659, 26.0028541,
                                               25.93875589, 25.8740586, 25.80615134, 25.74497627, 25.6817803,
                                               25.61816119, 25.55529656, 25.49149181, 25.43560961, 25.36694893,
                                               25.30741158])}
default_tip3p_weights = {'fn': np.array([0., -320.02143681, -638.70677524, -956.56156227, -1273.86888852, -1590.72017276,
                                        -1907.28358627, -2223.55298761, -2539.5609427 , -2855.37064859, -3171.01440754,
                                        -3486.51267573, -3801.87158594, -4117.15647017, -4432.22414904, -4747.21631242,
                                        -5062.11863508, -5376.90568794, -5691.62320312, -6006.29427367, -6320.8426477]),
                         'volume': np.array([26.96368168, 26.8888173, 26.82517038, 26.75463429, 26.69137309, 26.62797965,
                                            26.56274047, 26.50052355, 26.4397603, 26.38748205, 26.32390224, 26.26551186,
                                            26.20956075, 26.15009628, 26.09822373, 26.04674886, 25.99790136,
                                            25.94167758, 25.89578637, 25.84623103, 25.8039804])}

class Salinator(object):
    """
    A user-friendly wrapper for performing constant-salt-concentration simulations using the saltswap machinery.
    Use this object to neutralize a system with the saltswap-specific ion topologies, initialize the ion concentration,
    and perform the NCMC accelerated insertiona and deletion of salt.

    This class uses the Joung and Cheatham ion parameters, which are hard coded in the driver for the Swapper class.

    References
    ----------
    [1] Frenkel and Smit, Understanding Molecular Simulation, from algorithms to applications, second edition, 2002
    Academic Press. (Chapter 9, page 225 to 231)
    [2] Nilmeir, Crooks, Minh, Chodera, Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium
    simulation, PNAS, 108, E1009
    [3] Joung, Cheatham, J. Phys. Chem. B, Vol. 112, No. 30. (1 July 2008), pp. 9020-9041

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

    It's strongly adviced that the system is minimized and thermalized before running a production simulation. For the
    purpose of this example, we'll skip this and jump straight into Running a short simulation. This is achieved by
    mixing MD and MCMC saltswap moves.
    >>> for i in range(100):
    >>> ...     langevin.step(2000)
    >>> ...     salinator.update(nattempts=1)

    One can use to tools in saltswap.record to save the simulation data at a given iteration.

    """

    def __init__(self, context, system, topology, ncmc_integrator, salt_concentration, pressure, temperature,
                 npert=2000, nprop=5, water_name="HOH", cation_name='Na+', anion_name='Cl-', calibration_weights=None):
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
        pressure: simtk.unit.Quantity compatible with atmospheres, optional, default=None
            For explicit solvent simulations, the pressure.
        npert: int
            Number of ncmc perturbation kernels. Set to 1 for instantaneous switching
        nprop: int
            The number of propagation steps per perturbation.
        water_name = str, optional, default='HOH'
            Name of water residue that will be exchanged with salt
        cation_name: str, optional, default='Na+'
            Name of cation residue from which parameters are to be taken.
        anion_name: str, optional, default='Cl-'
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
            self.nprop = nprop
        elif self.npert == 1:
            # No propagation steps per perturbation, which mean no NCMC, ansd instantaneous insertion/deletion attempts.
            self.nprop = 0
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

    def add_salt(self, nsalt):
        """
        Add a specified amount of salt to the system.

        Parameters
        ----------
        nsalt: int
            the number of Na+ and Cl- pairs that will be added.
        """
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

    def neutralize(self):
        """
        Neutralize the system by adding counter-ions which have the topology required by saltswap.
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

        # Add the salt.
        self.add_salt(nsalt)

    def update(self, nattempts=None, chemical_potential=None, saltmax=None):
        """
        Perform MCMC salt insertion/deletion moves.
        """
        self.swapper.update(self.context, nattempts=nattempts, cost=chemical_potential, saltmax=saltmax)


class SAMSSalinator(Salinator):
    """
    Class to perform self-adjusted mixture sampling over salt-pair numbers between a minimum and maximum number. This
    class works in very much the same way as Salinator, except that one does not specify a macroscopic salt
    concentration, but instead specifies a range of salt occupancies that will be sampled in proportion to according to
    the specified target weights.

    Use this class for calculating the free energies to add an remove many numbers of salt pairs.

    Example
    -------
    Running SAMS on a box of TIP3P water. This sort of simulations that be used to calibrate the chemical potential for
    the TIP3P water model and a particular set of non-bonded parameters.

    >>> from simtk import openmm, unit
    >>> from openmmtools import testsystems
    >>> from openmmtools import integrators
    >>> from saltswap.record import Record

    Set the thermondynamic parameters
    >>> temperature = 300.0 * unit.kelvin
    >>> pressure = 1.0 * unit.atmospheres
    >>> salt_concentration = 0.2 * unit.molar

    Extract the test system:
    >>> testobj = getattr(testsystems, 'WaterBox')
    >>> testsys = testobj(model='tip3p', box_edge=30, nonbondedMethod=app.PME, cutoff=10*unit.angstrom,
    ...                   ewaldErrorTolerance=1E-4)
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

    Specify the maximum and minimum number of salt pairs that will be sampled over.
    >>> saltmax = 20
    >>> saltmin = 0

    The total number of states that will be sampled from is
    >>> nstates = saltmax - saltmin + 1

    Create the SAMS salinator:
    >>> sams_salinator = SAMSSalinator(saltmin, saltmax, context=context, system=testsys.system,
    >>> ...                            topology=testsys.topology, ncmc_integrator=ncmc_langevin,
    >>> ...                            salt_concentration=0.1 * unit.molar, pressure=pressure,
    >>> ...                            temperature=temperature, npert=1000, nprop=10, water_name='HOH')

    Although a salt concentration is specified, this will be ignored, and the above (by default) will attempt to sample
    uniformly over all salt pairs between 0 and 20 inclusive.

    It is important to note that SAMSSalinator uses the SAMS binary update scheme, which can be _very_ slow to
    converge if the initial bias is far from the negative of the true free energy of the salt numbers. SAMSalinator
    can be supplied with initial values of the biases via the 'initial_bias' flag. It's use is strongly recommended for
    production simulations.

    Initialize a netcdf file to store the simulation data, making sure to add the special commands to record the SAMS
    weights.
    >>> creator = Record.CreateNetCDF('out.nc')
    >>> ncfile = creator.create_netcdf(sams_salinator.swapper, nstates=nstates)

    Runing a short simulation by mixing MD and MCMC saltswap moves and saving the simulation data.
    >>> for i in range(100):
    >>> ...     langevin.step(2000)
    >>> ...     sams_salinator.update()
    >>> ...     Record.record_netcdf('out.nc', context, sams_salinator.swapper, i,sams_bias=sams_salinator.bias)

    All the data required to calibrate the chemical potential can be found in 'out.nc'.
    """
    def __init__(self, saltmin=0, saltmax=20, initial_bias=None, two_stage=True, beta=0.7, target_weights=None,
                 precision=0.1, **kwargs):
        """
        Parameters
        ----------
        saltmin: int
            the minimum number of salt pairs that will be sampled over.
        saltmax: int
            the maximum number of salt pairs that will be sampled over.
        initial_bias: numpy.ndarray
            array of the initial values of the SAMS biases.
        two_stage: bool
            whether to perform a two-stage procedure, where the first is a burn-in.
        beta: float
            the exponent of the gain in the two-stage procedure. Should be between 0.5 and 1.
        target_weights: numpy.ndarray
            the target state sampling proportions that the biases will be optimized to obtain.
        precision: float
            the precision to which the sampling proportions will match the target weights before the two-stage
            procedure is terminated.

        """
        if saltmax <= saltmin:
            raise Exception('The maximum amount of salt must be less than the minimum.')

        super(SAMSSalinator, self).__init__(**kwargs)

        self.saltmin = saltmin
        self.saltmax = saltmax
        self.nstates = 1 + saltmax - saltmin
        self.state_counts = np.zeros(self.nstates)
        self.nsalt = self.count_salt()

        if initial_bias is None:
            self.bias = np.zeros(self.nstates)
        else:
            self.bias = initial_bias

        self.adaptor = SAMSAdaptor(self.nstates, zetas=self.bias, target_weights=target_weights, two_stage=two_stage,
                                   beta=beta, precision=precision)
    def count_salt(self):
        """
        Count and return the number of neutral anion and cation pairs.
        """

        # Using the Swapper class that's inherited from the Salinator class.
        nwater, ncation, nanion = self.swapper.get_identity_counts()
        nsalt = min(ncation, nanion)
        return nsalt

    def generate_penalty(self):
        """
        Based on the current state and current value of the free energy estimate, calculate the penalty that will be
        used in the acceptance test to add or remove salt.

        Return
        ------
        penalty: list of floats.
            The bias that will be applied when inserting or deleting salt.
        """

        if self.nsalt == self.saltmax:
            penalty = [0.0, self.bias[self.nsalt - 1] - self.bias[self.nsalt]]
        elif self.nsalt <= self.saltmin:
            penalty = [self.bias[self.nsalt + 1] - self.bias[self.nsalt], 0.0]
        else:
            penalty = [self.bias[self.nsalt + 1] - self.bias[self.nsalt],
                       self.bias[self.nsalt - 1] - self.bias[self.nsalt]]
        return penalty

    def update(self):
        """
        Perform one satl insertion/deletion moves and update the estimate for the free energy.
        """
        # Insert or delete salt
        penalty = self.generate_penalty()
        self.swapper.attempt_identity_swap(self.context, penalty=penalty, saltmax=self.saltmax, saltmin=self.saltmin)

        # Update the SAMS bias
        self.nsalt = self.count_salt()
        noisy = np.zeros(self.nstates)
        noisy[self.nsalt] = 1
        self.state_counts[self.nsalt] += 1
        self.bias = self.adaptor.update(state=self.nsalt, noisy_observation=noisy, histogram=self.state_counts)


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
