#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""
Constant salt dynamics in OpenMM.

Description
-----------

This class implements a pure python 'constant salt concentration' functionality in OpenMM. In `constant salt
concentration simulations, the average concentration of salt in a saline reservior is fixed, but the reservoir is allowed
to exchange water and salt with the simulation system. This means the number of anions and cations in a simulation is a
fluctuating quantity. The constant salt concentration simulation is achieved using the semi grand canonical ensemble,
which allows molecules/particles to change identity.

Non-equilibrium candidate Monte Carlo (_ncmc) is used to increase acceptance rates of switching.

Based on code from openmm-constph.


Notes
-----

    * The code is still in development.
    * The Swapper class only performs moves that exchange two  water molecules for an anion-cation pair.
    * Swapper can be combined with molecular dynamics by alternating blocks of Swapper moves and molecular dynamics
     integration steps.

References
----------

[1] Frenkel and Smit, Understanding Molecular Simulation, from algorithms to applications, second edition, 2002 Academic Press.
    (Chapter 9, page 225 to 231)
[2] Nilmeir, Crooks, Minh, Chodera, Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation,PNAS,108,E1009

Example
-------

The OpenMM wrapper for Swapper is contained in the MCMCSampler class, which allows alternating steps of molecular
dynamics and Swapper moves. To run Swapper without that wrapper, see below for an example using a box of water.

>>> from simtk import openmm, unit
>>> from openmmtools.testsystems import WaterBox
>>> from openmmtools import integrators

Create the system.
>>> wbox = WaterBox()
>>> wbox.system.addForce(openmm.MonteCarloBarostat(1.*unit.atmospheres, 300.*unit.kelvin))

To perform NCMC, you must use a particular type of integrator that can accumulate the protocol work internally as well
as a custom integrator.
>>> equilibrium_integrator = integrators.LangevinIntegrator(temperature=300.*unit.kelvin)
>>> ncmc_integrator = integrators.ExternalPerturbationLangevinIntegrator(temperature=300.*unit.kelvin)
>>> compound_integrator = openmm.CompoundIntegrator()
>>> compound_integrator.addIntegrator(equilibrium_integrator)
>>> compound_integrator.addIntegrator(ncmc_integrator)

Create the context
>>> context = openmm.Context(wbox.system, compound_integrator)
>>> context.setPositions(wbox.positions)
>>> context.setVelocitiesToTemperature(300.*unit.kelvin)

Initialize the driver for performing salt-water exchanges
>>> swapper = Swapper(system=wbox,topology=wbox.topology,temperature=300.*unit.kelvin, delta_chem=317.0,
>>>...                ncmc_integrator=ncmc_integrator, pressure=1.*unit.atmospheres)

To perform constant salt concentration molecular dynamics, iterate between equilibrium dynamics and salt interstions.
For the purpose of this docstring the snippet below demonstrates a super short simulation.
>>> for iteration in range(5):
>>>...  compound_integrator.step(1)
>>>...  swapper.update(context,nattempts=1)

TODO
----
    * Read in ion parameters from system and topology. Currently, it is assumed the system is neutral with no cations
    or anions present. The ion parameters are currently supplied internally.

Copyright and license
---------------------

@author Gregory A. Ross <gregoryross.uk@gmail.com>

"""
from __future__ import print_function
import math
import random
import numpy as np
import simtk.unit as units
from .integrators import GHMCIntegrator, NCMCMetpropolizedGeodesicBAOAB
from openmmtools.integrators import ExternalPerturbationLangevinIntegrator

# MODULE CONSTANTS
kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
kB = kB.in_units_of(units.kilojoule_per_mole / units.kelvin)


def strip_in_unit_system(quant, unit_system=units.md_unit_system, compatible_with=None):
    """
    Strips the unit from a simtk.units.Quantity object and returns it's value conforming to a unit system
    Parameters
    ----------
    quant : simtk.unit.Quantity
        object from which units are to be stripped
    unit_system : simtk.unit.UnitSystem:
        unit system to which the unit needs to be converted, default is the OpenMM unit system (md_unit_system)
    compatible_with : simtk.unit.Unit
        Supply to make sure that the unit is compatible with an expected unit
    Returns
    -------
    quant : object with no units attached
    """
    if units.is_quantity(quant):
        if compatible_with is not None:
            quant = quant.in_units_of(compatible_with)
        return quant.value_in_unit_system(unit_system)
    else:
        return quant


class Swapper(object):
    """
    Monte Carlo driver for semi-grand canonical ensemble of swapping water molecules with ion cation pairs.

    Class that allows for particles and/or molecules to change identities and forcefield.

    """

    def __init__(self, system, topology, temperature, delta_chem, ncmc_integrator=None, pressure=None, nattempts_per_update=1,
                 npert=1, nprop=0, work_measurement='internal', waterName="HOH", cationName='Na+', anionName='Cl-'):
        """
        Initialize a Monte Carlo titration driver for semi-grand ensemble simulation.

        Parameters
        ----------
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        topology : simtk.openmm.app.topology
             Topology of the system
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature to be simulated.
        delta_chem : float or unit.Quantity
            The difference in chemical potential for swapping 2 water molecules for Na Cl.
            If it is a float, it is assumed to be in units of kT.
        ncmc_integrator : simtk.openmm.integrator
            The integrator used for NCMC propagation of insertion and deletions
        pressure : simtk.unit.Quantity compatible with atmospheres, optional, default=None
            For explicit solvent simulations, the pressure.
        npert : integer
            Number of ncmc perturbation kernels. Set to 1 for instantaneous switching
        nprop : integer
            Number of propagation kernels (MD steps) per ncmc perturbation kernel. Set to 0 for instantaneous switching
        work_measurement : str
            The name of method used to calculate the work of the NCMC protocol
        ncmc_timestep : simtk.unit.Quantity with units compatible with femtoseconds
            Timestep to use for ncmc switching
        waterName = str, optional, default='HOH'
            Name of water residue that will be exchanged with salt
        cationName : str, optional, default='Na+'
            Name of cation residue from which parameters are to be taken.
        anionName : str, optional, default='Cl-'
            Name of anion residue from which parameters are to be taken.
        """

        # Set defaults.

        # Store parameters.
        self.system = system
        self.topology = topology
        self.temperature = temperature
        self.kT = self.temperature * kB
        self.kT_unitless = strip_in_unit_system(kB) * strip_in_unit_system(temperature)  # in units of kJ/mol
        self.pressure = pressure
        self.delta_chem = delta_chem
        self.anionName = anionName
        self.cationName = cationName
        self.waterName = waterName
        self.ncmc_integrator = ncmc_integrator

        work_method_list = ['internal', 'external']
        if work_measurement in work_method_list:
            self.work_measurement = work_measurement
        elif work_measurement not in work_method_list and npert == 0:
            pass
        else:
            raise Exception('Method to calculate NCMC work, "{0}", not in supported list {1}'.format(work_measurement,
                                                                                                     work_method_list))
        # Saving the NCMC parameters
        self.npert = npert
        self.nprop = nprop

        # Store force object pointer.
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if force.__class__.__name__ == 'NonbondedForce':
                self.forces_to_update = force

        # Record the forces that need to be swicthed off for ncmc
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in
                  range(system.getNumForces())}
        # Control center mass remover
        if 'CMMotionRemover' in forces:
            self.cm_remover = forces['CMMotionRemover']
            self.cm_remover_freq = self.cm_remover.getFrequency()
        else:
            self.cm_remover = None
            self.cm_remover_freq = None
        # Check that system has MonteCarloBarostat if pressure is specified
        if pressure is not None:
            if 'MonteCarloBarostat' not in forces:
                raise Exception("`pressure` is specified, but `system` object lacks a `MonteCarloBarostat`")
            else:
                self.barostat = forces['MonteCarloBarostat']
                self.barofreq = self.barostat.getFrequency()
        else:
            self.barostat = None
            self.barofreq = None

        self.mutable_residues = self._identify_residues(self.topology,
                                                        residue_names=(self.waterName, self.anionName, self.cationName))

        # Describing the identities of water and ions with numpy vectors
        self.stateVector = self._initialize_state_vector()
        self.water_parameters = self._retrieve_residue_parameters(self.topology, self.waterName)
        self.cation_parameters = self._initialize_ion_parameters(ion_name=self.cationName, ion_params=None)
        self.anion_parameters = self._initialize_ion_parameters(ion_name=self.anionName, ion_params=None)

        # Setting the perturbation pathway for
        self._set_parampath()

        # Store list of exceptions that may need to be modified.
        self.nattempts_per_update = nattempts_per_update

        # Reset statistics.
        self.reset_statistics()
        self.naccepted_ncmc_integrator = []

        # Record the log acceptance probability for each proposal
        self.log_accept = []

        # Recording the cumulative work with the corresponding proposal
        self.proposal = [0, 0]      # [initial number of salt, proposed number of salt]
        self.cumulative_work = np.zeros(self.npert + 1)  # The work that corresponds to the above proposal

        # For counting the number of NaNs I get in ncmc. These are automatically rejected.
        self.nan = 0

    def _set_parampath(self, lj_step=1):
        """
        Produce an interpolation between the non-bonded forcefield parameters of water and ion parameters, with the
        option to perturb the partial charges more slowly than the Lennard-Jones (LJ) parameters. The interpolation serves
        as a path for ncmc.

        The perturbation of LJ parameters is slowed relative to the partial charge perturbation by introducing a lag in
        the update of the LJ path, which is specified by lj_step.  With lj_step > 1, more of the ncmc protocol is spent
        on the partial charge parameters.

        Parameters
        ---------
        lj_step : int
          the number of perturbation steps taken for the Lennard-Jones parameters per partial charge perturbation.

        """

        wat_params = self.water_parameters
        cat_params = self.cation_parameters
        an_params = self.anion_parameters

        self.wat2cat_parampath = []
        self.wat2an_parampath = []
        self.cat2wat_parampath = []
        self.an2wat_parampath = []

        # Pre-assigment of the data structures to store the perturbation path
        for atm_ind in range(len(wat_params)):
            self.wat2cat_parampath.append({'charge': [], 'sigma': [], 'epsilon': []})
            self.wat2an_parampath.append({'charge': [], 'sigma': [], 'epsilon': []})
            self.cat2wat_parampath.append({'charge': [], 'sigma': [], 'epsilon': []})
            self.an2wat_parampath.append({'charge': [], 'sigma': [], 'epsilon': []})

        # Specifying the effective number of perturbations for the Lennard-Jones parameters.
        if lj_step >= self.npert:
            # Ensuring logical consistency. The lag in the LJ perturbation cannot more than the number of perturbations.
            lj_step = 1
            npert_lj = self.npert
        else:
            if self.npert % lj_step == 0:
                npert_lj = np.ceil(float(self.npert) / float(lj_step))
            else:
                npert_lj = np.floor(float(self.npert) / float(lj_step))
        # For each atom in the water model (indexed by atm_ind), the parameters are linearly interpolated between the ions.
        # Both the forward and reverse directions (ie wat2cat and cat2wat) are calculated to save time at each ncmc perturbation
        n_lj = 0
        for n in range(self.npert + 1):
            frac_charge = float(n) / float(self.npert)
            if n % lj_step == 0:
                frac_lj = float(n_lj) / float(npert_lj)
                n_lj += 1
            for atm_ind in range(len(wat_params)):
                self.wat2cat_parampath[atm_ind]['charge'].append(
                    (1 - frac_charge) * wat_params[atm_ind]['charge'] + frac_charge * cat_params[atm_ind]['charge'])
                self.wat2an_parampath[atm_ind]['charge'].append(
                    (1 - frac_charge) * wat_params[atm_ind]['charge'] + frac_charge * an_params[atm_ind]['charge'])
                self.an2wat_parampath[atm_ind]['charge'].append(
                    (1 - frac_charge) * an_params[atm_ind]['charge'] + frac_charge * wat_params[atm_ind]['charge'])
                self.cat2wat_parampath[atm_ind]['charge'].append(
                    (1 - frac_charge) * cat_params[atm_ind]['charge'] + frac_charge * wat_params[atm_ind]['charge'])
                for type in ['sigma', 'epsilon']:
                    self.wat2cat_parampath[atm_ind][type].append(
                        (1 - frac_lj) * wat_params[atm_ind][type] + frac_lj * cat_params[atm_ind][type])
                    self.wat2an_parampath[atm_ind][type].append(
                        (1 - frac_lj) * wat_params[atm_ind][type] + frac_lj * an_params[atm_ind][type])
                    self.an2wat_parampath[atm_ind][type].append(
                        (1 - frac_lj) * an_params[atm_ind][type] + frac_lj * wat_params[atm_ind][type])
                    self.cat2wat_parampath[atm_ind][type].append(
                        (1 - frac_lj) * cat_params[atm_ind][type] + frac_lj * wat_params[atm_ind][type])

    def _retrieve_residue_parameters(self, topology, resname):
        """
        Retrieves the non-bonded parameters for a specified residue.

        Parameters
        ----------
        topology : simtk.openmm.app.topology
            The topology from which water residues are to be identified.
        resname : str
            The residue name of the residue from which parameters are to be retrieved.

        Returns
        -------
        param_list : list of dict of str:float
            List of NonbondedForce parameter dict ('charge', 'sigma', 'epsilon') for each atom.

        """

        param_list = []
        for residue in topology.residues():
            if residue.name == resname:
                atoms = [atom for atom in residue.atoms()]
                for atm in atoms:
                    [charge, sigma, epsilon] = self.forces_to_update.getParticleParameters(atm.index)
                    parameters = {'charge': strip_in_unit_system(charge), 'sigma': strip_in_unit_system(sigma),
                                  'epsilon': strip_in_unit_system(epsilon)}
                    param_list.append(parameters)
                return param_list
        raise Exception("resname '%s' not found in topology" % resname)

    def _initialize_ion_parameters(self, ion_name, ion_params=None):
        """
        Initialize the set of ion non-bonded parameters so that they match the number of atoms of the water model.

        Parameters
        ----------
        water_name : str
            The residue name of the water molecule
        ion_name : str
            The residue name of the ion
        ion_params : dict of str:float
            NonbondedForce parameter dict ('charge', 'sigma', 'epsilon') for ion.
        Returns
        -------
        """

        # Creating a list of non-bonded parameters that matches the size of the water model.
        num_wat_atoms = len(self.water_parameters)

        # Initialising dummy atoms to have no non-bonded interactions
        # eps = sys.float_info.epsilon       # the smallest float that's not zero
        eps = 0.0
        ion_param_list = num_wat_atoms * [{'charge': eps, 'sigma': eps, 'epsilon': eps}]
        # Making the first element of list of parameter dictionaries the ion. This means that ions will be centered
        # on the water oxygen atoms.
        # If ion parameters are not supplied, use Joung and Cheatham parameters.
        if ion_name == self.cationName:
            if ion_params is None:
                ion_param_list[0] = {'charge': 1.0, 'sigma': 0.2439281, 'epsilon': 0.0874393}
            else:
                ion_param_list[0] = ion_params
        elif ion_name == self.anionName:
            if ion_params is None:
                ion_param_list[0] = {'charge': -1.0, 'sigma': 0.4477657, 'epsilon': 0.0355910}
            else:
                ion_param_list[0] = ion_params
        else:
            raise NameError('Ion name %s does not match known cation or anion name' % ion_name)

        return ion_param_list

    def _identify_residues(self, topology, residue_names):
        """
        Compile a list of residues that could be converted to/from another chemical species.

        Parameters
        ----------
        topology : simtk.openmm.app.topology
            The topology from which water residues are to be identified.
        residue_names : list of str
            Residues identified as water molecules.

        Returns
        -------
        water_residues : list of simtk.openmm.app.Residue
            Water residues.
        """
        target_residues = list()
        for residue in topology.residues():
            if residue.name in residue_names:
                target_residues.append(residue)

        return target_residues

    def _initialize_state_vector(self):
        """
        Stores the identity of the mutabable residues in a numpy array for efficient seaching and updating of
        residue identies.

        Returns
        -------
        stateVector : numpy array
            Array of 0s, 1s, and 2s to indicate water, sodium, and chlorine.

        """
        names = [res.name for res in self.mutable_residues]
        stateVector = np.zeros(len(names))
        for i in range(len(names)):
            if names[i] == self.waterName:
                stateVector[i] = 0
            elif names[i] == self.cationName:
                stateVector[i] = 1
            elif names[i] == self.anionName:
                stateVector[i] = 2
        return stateVector

    def compare_protocol_work(self, context):
        """
        Test implementation of NCMC to compare two different methods of calculating protocol work.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        """
        # Randomly pick waters to turn into salt
        mode='add salt'
        exchange_indices= np.random.choice(a=np.where(self.stateVector == 0)[0], size=2, replace=False)

        if self.cm_remover is not None:
            self.cm_remover.setFrequency(0)

        # Reset protocol work
        self.ncmc_integrator.setGlobalVariableByName("first_step", 0)

        internal_work = np.zeros(self.npert + 1)
        external_work = np.zeros(self.npert + 1)
        ext_wrk = 0.0
        # Propagation
        self.ncmc_integrator.step(1)
        for stage in range(self.npert + 1):
            # Energy before perturbation
            pot_initial = self._get_potential_energy(context)
            # Perturbation
            self._update_forces(mode, exchange_indices, stage)
            self.forces_to_update.updateParametersInContext(context)
            pot_final = self._get_potential_energy(context)
            # Energy after perturbation
            ext_wrk += (pot_final - pot_initial) / self.kT
            external_work[stage] = ext_wrk
            # Propagation
            self.ncmc_integrator.step(1)
            internal_work[stage] = self.ncmc_integrator.get_protocol_work(dimensionless=True)

        # Re-instate center of mass motion if on.
        if self.cm_remover is not None:
            self.cm_remover.setFrequency(self.cm_remover_freq)

        # Return the salt that has been added.
        self._update_forces('remove salt', exchange_indices, stage=self.npert)
        self.forces_to_update.updateParametersInContext(context)

        # I don't care about restoring the positions and velocities as this is a test.

        return internal_work, external_work

    def _ncmc(self, context, npert, nprop, mode, exchange_indices, work_measurement='internal'):
        """
        Updates the context with either inserted or deleted salt using non-equilibrium candidate Monte Carlo.

        So that the protocol is time symmetric, the protocol is given by
             propagation -> perturbation -> propagation


        WARNING: The velocity Verlet integrator is depracted.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        npert : int
            The number of ncmc perturbation-propagation kernels to use.
        nprop : int
            The number of propagation steps per perturbation kernel
        mode : string
            Either 'add salt' or 'remove  salt', which is passed to '_update_forces'
        exchange_indices : numpy array
            Two element vector containing the residue indices that have been changed
        work_measurement : str
            The method used to calculate the protocol work of the NCMC propagator.

        Returns
        -------
        work: float
            The work for appropriate for the stated propagator in units of KT.
        cumulative_work: float
            The cumulative protocol work for each ncmc step

        """
        if self.cm_remover is not None:
            self.cm_remover.setFrequency(0)

        cumulative_work = np.zeros(npert + 1)

        if work_measurement == 'internal':
            self.ncmc_integrator.setGlobalVariableByName("first_step", 0)
            # Propagation
            self.ncmc_integrator.step(1)
            for stage in range(npert + 1):
                # Perturbation
                self._update_forces(mode, exchange_indices, stage)
                self.forces_to_update.updateParametersInContext(context)
                # Propagation
                self.ncmc_integrator.step(1)
                cumulative_work[stage] = self.ncmc_integrator.get_protocol_work(dimensionless=True)

            # Extract the internally calculated work from the integrator
            #work = ncmc_integrator.getGlobalVariableByName('protocol_work') / self.kT_unitless
            work = self.ncmc_integrator.get_protocol_work(dimensionless=True)

            # Save the acceptance rate for the ncmc protocol if the propagator is Metropolized.
            try:
                self.naccepted_ncmc_integrator.append(self.ncmc_integrator.getGlobalVariableByName('naccept') / self.ncmc_integrator.getGlobalVariableByName('ntrials'))
            except:
                self.naccepted_ncmc_integrator.append(0.0)

        elif work_measurement == 'external':
            # Like the GHMC integrator above, except that energies are calculated with _get_potential_energy() for
            # testing and benchmarking. Option came about due to errors in the energy calculations with CustomIntegrator
            # that have since been fixed.

            work = 0.0  # Unitless work
            # Propagation
            self.ncmc_integrator.step(nprop)
            for stage in range(npert + 1):
                # Getting the potential energy before the perturbation
                pot_initial = self._get_potential_energy(context)
                # Perturbation
                self._update_forces(mode, exchange_indices, stage)
                self.forces_to_update.updateParametersInContext(context)
                # Getting the potential energy after the perturbation
                pot_final = self._get_potential_energy(context)
                # Propagation
                self.ncmc_integrator.step(nprop)
                # Update the accumulated work
                work += (pot_final - pot_initial) / self.kT
                cumulative_work[stage] = work

            # Save the acceptance rate for the ncmc protocol
            try:
                self.naccepted_ncmc_integrator.append(
                    self.ncmc_integrator.getGlobalVariableByName('naccept') / ncmc_integrator.getGlobalVariableByName('ntrials'))
            except:
                self.naccepted_ncmc_integrator.append(0.0)

        else:
            raise Exception('Method to calculate work, "{0}", not recognized'.format(work_measurement))

        #self.ncmc_integrator.setCurrentIntegrator(0)

        if self.cm_remover is not None:
            self.cm_remover.setFrequency(self.cm_remover_freq)

        return work, cumulative_work

    def attempt_identity_swap(self, context, penalty, saltmax=None):
        """
        Attempt the exchange of (possibly multiple) chemical species.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        penalty : list floats
            The free energy to add salt (first index) and remove salt (second index)
        saltmax : int
            The maximum number of salt pairs that you wish to be added. If None, then the maximum number is the
            number of water molecules divided by 2.
        """
        self.nattempted += 1

        if type(penalty) == float:
            penalty = [penalty, -penalty]

        # If using ncmc, store initial positions.
        if self.nprop > 0:
            state = context.getState(getPositions=True, getVelocities=True)
            initial_positions = state.getPositions()
            initial_box_vectors = state.getPeriodicBoxVectors()
            initial_velocities = state.getVelocities()

        # Introducing a maximum capacity of salt molecules for the 'self adjusted mixture sampling calibration.
        if saltmax == None:
            saltmax = (len(self.mutable_residues) - len(self.mutable_residues) % 2) / 2

        # Initializing the exponent of the acceptance test. Adding to it as we go along.
        log_accept = 0.0

        # Whether to delete or add salt by selecting random water molecules to turn into a cation and an anion or vice versa.
        nwats, ncation, nanion = self.get_identity_counts()
        if (ncation == 0):
            change_indices = np.random.choice(a=np.where(self.stateVector == 0)[0], size=2, replace=False)
            mode = "add salt"
            log_accept -= np.log(2)  # Due to asymmetric proposal probabilities
            cost = penalty[0]  # The free energy to remove salt and add 2 waters to bulk water
        elif (ncation >= saltmax):
            mode = "remove salt"
            cation_index = np.random.choice(a=np.where(self.stateVector == 1)[0], size=1)[0]
            anion_index = np.random.choice(a=np.where(self.stateVector == 2)[0], size=1)[0]
            change_indices = np.array([cation_index, anion_index])
            log_accept -= np.log(2)  # Due to asymmetric proposal probabilities
            cost = penalty[1]
        elif (np.random.random() < 0.5):
            change_indices = np.random.choice(a=np.where(self.stateVector == 0)[0], size=2, replace=False)
            mode = "add salt"
            cost = penalty[0]
        else:
            mode = "remove salt"
            cation_index = np.random.choice(a=np.where(self.stateVector == 1)[0], size=1)[0]
            anion_index = np.random.choice(a=np.where(self.stateVector == 2)[0], size=1)[0]
            change_indices = np.array([cation_index, anion_index])
            cost = penalty[1]

        # Perform perturbation to remove or add salt with ncmc and calculate energies
        if self.nprop > 0:
            try:
                work, cumulative_work = self._ncmc(context, self.npert, self.nprop, mode, change_indices,
                                                   work_measurement=self.work_measurement)
            except Exception as detail:
                work = np.inf  # If the simulation explodes during ncmc, reject with high work
                cumulative_work = 0.0
                print(detail)
                if detail[0] == 'Particle coordinate is nan':
                    self.nan += 1
                else:
                    print(detail)
        # Else make an instantaneous insertion or deletion.
        else:
            pot_initial = self._get_potential_energy(context)
            self._update_forces(mode, change_indices, stage=self.npert)
            self.forces_to_update.updateParametersInContext(context)
            pot_final = self._get_potential_energy(context)
            work = (pot_final - pot_initial) / self.kT
            cumulative_work = 0.0

        # Saving the work and proposal (already in units of KT)
        if mode == "remove salt":
            self.proposal = [ncation, ncation - 1]
            self.cumulative_work = cumulative_work
        else:
            self.proposal = [ncation, ncation + 1]
            self.cumulative_work = cumulative_work

        # Cost = F_final - F_initial, where F_initial is the free energy to have the current number of salt molecules.
        log_accept += -cost - work
        # The acceptance test must include the probability of uniformally selecting which salt pair or water to exchange
        (nwats, ncation, nanion) = self.get_identity_counts()
        if mode == 'add salt':
            log_accept += np.log(1.0 * nwats * (nwats - 1) / (nanion + 1) / (nanion + 1))
        else:
            log_accept += np.log(1.0 * ncation * nanion / (nwats + 1) / (nwats + 2))

        # Record the log of the acceptance probability
        self.log_accept = log_accept

        # Accept or reject:
        if (log_accept > 0.0) or (random.random() < math.exp(log_accept)):
            # Accept :D
            self.naccepted += 1
            self._set_identity(mode, change_indices)
            if self.nprop > 0:
                context.setVelocities(-context.getState(getVelocities=True).getVelocities(asNumpy=True))
        else:
            # Reject :(
            # Revert parameters to their previous value
            self._update_forces(mode, change_indices, stage=0)
            self.forces_to_update.updateParametersInContext(context)
            if self.nprop > 0:
                context.setPositions(initial_positions)
                context.setVelocities(initial_velocities)
                context.setPeriodicBoxVectors(*initial_box_vectors)

    def _set_identity(self, mode, exchange_indices):
        """
        Function to set the names of the mutated residues and update the state vector. Called after a transformation
        of the forcefield parameters has been accepted.

        Parameters
        ----------
        mode : string
            Either 'add salt' or 'remove  salt'
        exchange_indices : numpy array
            Two element vector containing the residue indices that have been changed

        """

        if mode == "add salt":
            self.mutable_residues[exchange_indices[0]].name = self.cationName
            self.stateVector[exchange_indices[0]] = 1
            self.mutable_residues[exchange_indices[1]].name = self.anionName
            self.stateVector[exchange_indices[1]] = 2
        if mode == "remove salt":
            self.mutable_residues[exchange_indices[0]].name = self.waterName
            self.mutable_residues[exchange_indices[1]].name = self.waterName
            self.stateVector[exchange_indices] = 0

    def _update_forces(self, mode, exchange_indices, stage=0):
        """
        Update the forcefield parameters accoring depending on whether inserting salt or water. For inserting salt,
        2 water molecules

        Parameters
        ----------
        mode : string
            Whether the supplied indices will be used to 'add salt' or 'remove salt'
        exchange_indices : numpy array
            Indices of residues will be converted to cation and anion, or which cation and anion will be turned
            into 2 water residue.
        stage : int
            The index that points to the parameter value

        """
        if mode == 'add salt':
            molecule1 = [atom for atom in self.mutable_residues[exchange_indices[0]].atoms()]
            molecule2 = [atom for atom in self.mutable_residues[exchange_indices[1]].atoms()]
            atm_index = 0
            for atm1, atm2 in zip(molecule1, molecule2):
                self.forces_to_update.setParticleParameters(atm1.index,
                                                            charge=self.wat2cat_parampath[atm_index]['charge'][stage],
                                                            sigma=self.wat2cat_parampath[atm_index]['sigma'][stage],
                                                            epsilon=self.wat2cat_parampath[atm_index]['epsilon'][stage])
                self.forces_to_update.setParticleParameters(atm2.index,
                                                            charge=self.wat2an_parampath[atm_index]['charge'][stage],
                                                            sigma=self.wat2an_parampath[atm_index]['sigma'][stage],
                                                            epsilon=self.wat2an_parampath[atm_index]['epsilon'][stage])
                atm_index += 1
        if mode == 'remove salt':
            molecule1 = [atom for atom in self.mutable_residues[exchange_indices[0]].atoms()]
            molecule2 = [atom for atom in self.mutable_residues[exchange_indices[1]].atoms()]
            atm_index = 0
            for atm1, atm2 in zip(molecule1, molecule2):
                self.forces_to_update.setParticleParameters(atm1.index,
                                                            charge=self.cat2wat_parampath[atm_index]['charge'][stage],
                                                            sigma=self.cat2wat_parampath[atm_index]['sigma'][stage],
                                                            epsilon=self.cat2wat_parampath[atm_index]['epsilon'][stage])
                self.forces_to_update.setParticleParameters(atm2.index,
                                                            charge=self.an2wat_parampath[atm_index]['charge'][stage],
                                                            sigma=self.an2wat_parampath[atm_index]['sigma'][stage],
                                                            epsilon=self.an2wat_parampath[atm_index]['epsilon'][stage])
                atm_index += 1

    def _get_potential_energy(self, context):
        """
        Extract the potential energy of the system

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to get the energy of
        Returns
        -------
        potential energy : qunatity in default unit of energy

        """
        state = context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy()
        return pot_energy

    def _compute_log_probability(self, context):
        """
        Compute log probability of current configuration and protonation state.

        Parameters
        ----------

        context : simtk.openmm.Context
            the context

        Returns
        -------
        log_P : float
            log probability of the current context
        pot_energy : float
            potential energy of the current context
        kin_energy : float
            kinetic energy of the current context

        """

        # Add energetic contribution to log probability.
        state = context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy()
        kin_energy = state.getKineticEnergy()
        total_energy = pot_energy + kin_energy
        log_P = - total_energy / self.kT

        if self.pressure is not None:
            # Add pressure contribution for periodic simulations.
            volume = context.getState().getPeriodicBoxVolume()
            log_P += -self.pressure * volume * units.AVOGADRO_CONSTANT_NA / self.kT

        # Return the log probability.
        return log_P, pot_energy, kin_energy

    def update(self, context, nattempts=None, cost=None, saltmax=None):
        """
        Perform a number of Monte Carlo update trials for the titration state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        nattempts : integer
            Number of salt insertion and deletion moves to attempt.
        cost : float, int, or units.Quantity
            The difference in chemical potential of two water molecules and an anion and cation.
            If cost is a unit.Quantity, the cost must be in kJ/mol. If it is a float, it is assumed to be in units
            of kT.
        saltmax : int
            The maximum number of anion-cation pairs that can be inserted. If not specified, the maximum number is
            roughly half the total number of water molecules.

        """
        if nattempts == None:
            nattempts = self.nattempts_per_update

        # If no cost is supplied, use the supplied chemical potential
        if cost is None:
            cost = self.delta_chem

        # Check the type of the cost to ensure that it enters into the acceptance test as unitless quantity.
        if type(cost) == units.Quantity:
            # [free energy to add salt, free energy to remove salt]
            cost = [cost / self.kT, -cost / self.kT]
        elif type(cost) == float:
            # [free energy to add salt, free energy to remove salt]
            cost = [cost, -cost]
        elif type(cost) == int:
            # [free energy to add salt, free energy to remove salt]
            cost = [float(cost), -float(cost)]
        else:
            raise Exception('The data type of the chemical potential, "{0}", is not recognized'.format(cost))

        # Perform a number of trial salt insertion/deletion attempts.
        for attempt in range(nattempts):
            self.attempt_identity_swap(context, penalty=cost, saltmax=saltmax)
        return

    def get_acceptance_probability(self):
        """
        Return the fraction of accepted moves

        Returns
        -------

        fraction : float
            the fraction of accepted moves

        """
        return float(self.naccepted) / float(self.nattempted)

    def reset_statistics(self):
        """
        Reset statistics of insertion/deletion tracking.
        """

        self.nattempted = 0
        self.naccepted = 0

    def get_identity_counts(self):
        """
        Returns the total number of waters, cations, and anions

        Returns
        -------

        counts : tuple of integers
            The number of waters, cations, and anions respectively

        """
        nwats = np.sum(self.stateVector == 0)
        ncation = np.sum(self.stateVector == 1)
        nanion = np.sum(self.stateVector == 2)
        return (nwats, ncation, nanion)
