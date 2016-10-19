#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""
Constant salt dynamics in OpenMM.

Description
-----------

This module implements a pure python constant salt functionality in OpenMM, based on OpenMM's constant pH implementation.
Constant salt simulations are achieved using the semi grand canonical enemble, which molecules/particles to change indentity.

Based on code from openmm-constph.

Non-equilibrium candidate Monte Carlo (NCMC) can be used to increase acceptance rates of switching.

Notes
-----

The code is works, but is still in development.

References
----------

[1] Frenkel and Smit, Understanding Molecular Simulation, from algorithms to applications, second edition, 2002 Academic Press.
    (Chapter 9, page 225 to 231)
[2] Nilmeir, Crooks, Minh, Chodera, Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation,PNAS,108,E1009

Examples
--------

Coming soon to an interpreter near you!

TODO
----
    * Increase accepance rates by either using configuration biasing, or NCMC switching moves. The latter is preferred.
    * Check that acceptance test is correct; is proposal for exchanges accounted for?

Copyright and license
---------------------

@author Gregory A. Ross <gregoryross.uk@gmail.com>

"""
from __future__ import print_function
import sys
import math
import random
import numpy as np
#import simtk.openmm as openmm
import simtk.unit as units
#from openmmtools.integrators import VelocityVerletIntegrator

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

class SaltSwap(object):
    """
    Monte Carlo driver for semi-grand canonical ensemble of swapping water molecules with ion cation pairs.

    Class that allows for particles and/or molecules to change identities and forcefield.

    """

    def __init__(self, system, topology, temperature, delta_chem, integrator, pressure=None, nattempts_per_update=50, debug=False,
        npert=1, nprop=0, propagator='GHMC', waterName="HOH", cationName='Na+', anionName='Cl-'):
        """
        Initialize a Monte Carlo titration driver for semi-grand ensemble simulation.

        Parameters
        ----------
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature to be simulated.
        delta_chem : float
            The difference in chemical potential for swapping 2 water molecules for Na Cl.
        chemical_names : list of strings
            Names of each of the residues whose parameters will be exchanged.
        integrator : simtk.openmm.integrator
            The integrator used for dynamics outside of SaltSwap
        pressure : simtk.unit.Quantity compatible with atmospheres, optional, default=None
            For explicit solvent simulations, the pressure.
        debug : bool, optional, default=False
            Turn debug information on/off.
        npert : integer
            Number of NCMC perturbation kernels. Set to 1 for instantaneous switching
        nprop : integer
            Number of propagation kernels (MD steps) per NCMC perturbation kernel. Set to 0 for instantaneous switching
        ncmc_timestep : simtk.unit.Quantity with units compatible with femtoseconds
            Timestep to use for NCMC switching
        waterName = str, optional, default='HOH'
            Name of water residue that will be exchanged with salt
        cationName : str, optional, default='Na+'
            Name of cation residue from which parameters are to be taken.
        anionName : str, optional, default='Cl-'
            Name of anion residue from which parameters are to be taken.

        Todo
        ----
        """

        # Set defaults.

        # Store parameters.
        self.system = system
        self.topology = topology
        self.temperature = temperature
        self.kT = self.temperature*kB
        self.kT_unitless = strip_in_unit_system(kB)*strip_in_unit_system(temperature)      # in units of kJ/mol
        self.pressure = pressure
        self.delta_chem = delta_chem
        self.debug = debug

        self.anionName = anionName
        self.cationName = cationName
        self.waterName = waterName

        self.integrator = integrator

        proplist = ['GHMC', 'GHMC_old', 'velocityVerlet']
        if propagator in proplist:
            self.propagator = propagator
        elif propagator not in proplist and npert==0:
            pass
        else:
            raise Exception('NCMC propagator {0} not in supported list {1}'.format(propagator, proplist))

        self.npert = npert
        self.nprop = nprop

        # Store force object pointer.
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if force.__class__.__name__ == 'NonbondedForce':
                self.forces_to_update = force

        # Check that system has MonteCarloBarostat if pressure is specified.
        if pressure is not None:
            forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
            if 'MonteCarloBarostat' not in forces:
                self.barofreq = None
                raise Exception("`pressure` is specified, but `system` object lacks a `MonteCarloBarostat`")
            else:
                self.barostat = forces['MonteCarloBarostat']
                self.barofreq = self.barostat.getFrequency()

        self.mutable_residues = self.identifyResidues(self.topology,residue_names=(self.waterName,self.anionName,self.cationName))

        self.stateVector = self.initializeStateVector()
        self.water_parameters = self.retrieveResidueParameters(self.topology,self.waterName)
        self.cation_parameters = self.initializeIonParameters(ion_name=self.cationName,ion_params=None)
        self.anion_parameters = self.initializeIonParameters(ion_name=self.anionName,ion_params=None)

        # Setting the perturbation pathway for
        self.set_parampath()

        # Describing the identities of water and ions with numpy vectors

        # Track simulation state
        # self.kin_energies = units.Quantity(list(), units.kilocalorie_per_mole)
         #self.pot_energy = 0*units.kilocalorie_per_mole)

        # Store list of exceptions that may need to be modified.
        self.nattempts_per_update = nattempts_per_update

        # Reset statistics.
        self.resetStatistics()

        # For comparing NCMC and instance switching energies only:
        #self.nrg_ncmc = []
        #self.nrg_isnt = []

        # Saving the work values for adding and removing salt
        self.work_add = []
        self.work_rm = []
        self.naccepted_ghmc = []
        # For counting the number of NaNs I get in NCMC. These are automatically rejected.
        self.nan = 0
        return

    def set_parampath(self):
        """
        Produce a linear interpolation between the nonbonded forcefield parameters of water and ion parameters.
        The linear interpolation serves as a path for NCMC.

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
            self.wat2cat_parampath.append({'charge':[], 'sigma':[], 'epsilon':[]})
            self.wat2an_parampath.append({'charge':[], 'sigma':[], 'epsilon':[]})
            self.cat2wat_parampath.append({'charge':[], 'sigma':[], 'epsilon':[]})
            self.an2wat_parampath.append({'charge':[], 'sigma':[], 'epsilon':[]})

        # For each atom in the water model (indexed by atm_ind), the parameters are linearly interpolated between the ions.
        # Both the forward and reverse directions (ie wat2cat and cat2wat) are calculated to save time at each NCMC perturbation
        for n in range(self.npert):
            fraction = float(n + 1)/float(self.npert)
            for atm_ind in range(len(wat_params)):
                for type in ['charge','sigma','epsilon']:
                    self.wat2cat_parampath[atm_ind][type].append((1-fraction)*wat_params[atm_ind][type] + fraction*cat_params[atm_ind][type])
                    self.wat2an_parampath[atm_ind][type].append((1-fraction)*wat_params[atm_ind][type] + fraction*an_params[atm_ind][type])
                    self.an2wat_parampath[atm_ind][type].append((1-fraction)*an_params[atm_ind][type] + fraction*wat_params[atm_ind][type])
                    self.cat2wat_parampath[atm_ind][type].append((1-fraction)*cat_params[atm_ind][type] + fraction*wat_params[atm_ind][type])

    def retrieveResidueParameters(self, topology, resname):
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
                    parameters = {'charge': strip_in_unit_system(charge), 'sigma': strip_in_unit_system(sigma), 'epsilon': strip_in_unit_system(epsilon)}
                    param_list.append(parameters)
                return param_list
        raise Exception("resname '%s' not found in topology" % resname)

    def initializeIonParameters(self,ion_name,ion_params=None):
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

        # TODO: set eps to exactly zero to verify bug
        # Initialising dummy atoms to have no non-bonded interactions
        #eps = sys.float_info.epsilon       # the smallest float that's not zero
        eps = 0.0
        ion_param_list = num_wat_atoms*[{'charge': eps, 'sigma': eps, 'epsilon':eps}]
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
                ion_param_list[0] = {'charge': -1.0, 'sigma': 0.4477657, 'epsilon':0.0355910}
            else:
                ion_param_list[0] = ion_params
        else:
            raise NameError('Ion name %s does not match known cation or anion name' % ion_name)

        return  ion_param_list

    def identifyResidues(self, topology, residue_names):
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

        if self.debug: print('identifyResidues: %d %s molecules identified.' % (len(target_residues),residue_names[0]))
        return target_residues

    def initializeStateVector(self):
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
            if names[i] == self.waterName:  stateVector[i] = 0
            elif names[i] == self.cationName: stateVector[i] = 1
            elif names[i] == self.anionName: stateVector[i] = 2
        return stateVector


    def resetStatistics(self):
        """
        Reset statistics of insertion/deletion tracking.
        """

        self.nattempted = 0
        self.naccepted = 0

        return

    def attempt_identity_swap(self,context,penalty,saltmax=None):
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

        if type(penalty)==float:
            penalty = [penalty,-penalty]

        if self.barostat is not None:
            self.barostat.setFrequency(0)

        # If using NCMC, store initial positions.
        if self.nprop > 0:
            initial_positions = context.getState(getPositions=True).getPositions()
            initial_velocities = context.getState(getVelocities=True).getVelocities()

        # Introducing a maximum capacity of salt molecules for the 'self adjusted mixture sampling calibration.
        if saltmax == None:
            saltmax = (len(self.mutable_residues) - len(self.mutable_residues) % 2)/2

        # Initializing the exponent of the acceptance test. Adding to it as we go along.
        log_accept = 0.0
        # Whether to delete or add salt by selecting random water molecules to turn into a cation and an anion or vice versa.
        if (sum(self.stateVector==1) == 0):
            change_indices = np.random.choice(a=np.where(self.stateVector == 0)[0],size=2,replace=False)
            mode_forward = "add salt"
            mode_backward ="remove salt"
            log_accept -= np.log(2)                     # Due to asymmetric proposal probabilities
            cost = penalty[0]              # The free energy to remove salt and add 2 waters to bulk water
        elif (sum(self.stateVector==1) >= saltmax):
            mode_forward = "remove salt"
            mode_backward = "add salt"
            cation_index = np.random.choice(a=np.where(self.stateVector==1)[0],size=1)
            anion_index = np.random.choice(a=np.where(self.stateVector==2)[0],size=1)
            change_indices = np.array([cation_index,anion_index])
            log_accept -= np.log(2)                     # Due to asymmetric proposal probabilities
            cost = penalty[1]
        elif (np.random.random() < 0.5):
            change_indices = np.random.choice(a=np.where(self.stateVector == 0)[0],size=2,replace=False)
            mode_forward = "add salt"
            mode_backward ="remove salt"
            cost = penalty[0]
        else:
            mode_forward = "remove salt"
            mode_backward = "add salt"
            cation_index = np.random.choice(a=np.where(self.stateVector==1)[0],size=1)
            anion_index = np.random.choice(a=np.where(self.stateVector==2)[0],size=1)
            change_indices = np.array([cation_index,anion_index])
            cost = penalty[1]

        # Perform perturbation to remove or add salt with NCMC and calculate energies
        if self.nprop > 0:
            try:
                work = self.NCMC(context,self.npert,self.nprop,mode_forward,change_indices,propagator=self.propagator)
            except Exception as detail:
                work = 1000000000000.0               # If the simulation explodes during NCMC, reject with high work
                if detail[0]=='Particle coordinate is nan':
                    self.nan += 1
                else:
                    print(detail)
        else:
            # TODO: check whether stage parameter is correct
            pot_initial = self.getPotEnergy(context)
            self.updateForces(mode_forward,change_indices,stage=self.npert-1)
            self.forces_to_update.updateParametersInContext(context)
            pot_final= self.getPotEnergy(context)
            work = (pot_final - pot_initial)/self.kT

        # Computing the work (already in units of KT)
        if mode_forward == "remove salt":
            self.work_rm.append(work)
        else:
            self.work_add.append(work)

        # Cost = F_final - F_initial, where F_initial is the free energy to have the current number of salt molecules.
        log_accept += -cost - work
 
        # The acceptance test must include the probability of uniformally selecting which salt pair or water to exchange
        (nwats,ncation,nanion) = self.getIdentityCounts()
        if mode_forward == 'add salt' :
            log_accept += np.log(1.0*nwats*(nwats-1)/(nanion+1)/(nanion+1))
        else :
            log_accept += np.log(1.0*ncation*nanion/(nwats+1)/(nwats+2))

        # Accept or reject:
        if (log_accept > 0.0) or (random.random() < math.exp(log_accept)) :
            # Accept :D
            self.naccepted += 1
            self.setIdentity(mode_forward,change_indices)
            if self.nprop > 0:
                context.setVelocities(-context.getState(getVelocities=True).getVelocities(asNumpy=True))
        else:
            # Reject :(
            # Revert parameters to their previous value
            self.updateForces(mode_backward,change_indices,stage=self.npert-1)
            #self.updateForces_fractional(mode_backward,change_indices,fraction=1.0)    # The old way of reseting parameters
            self.forces_to_update.updateParametersInContext(context)
            if self.nprop > 0:
                context.setPositions(initial_positions)
                context.setVelocities(initial_velocities)

        if self.barostat is not None:
            self.barostat.setFrequency(self.barofreq)


    def NCMC(self,context, npert, nprop, mode, exchange_indices, propagator='GHMC'):
        """
        Performs nonequilibrium candidate Monte Carlo for the addition or removal of salt.
        So that the protocol is time symmetric, the protocol is given by
             propagation -> perturbation -> propagation


        TODO: have the propagation kernel type read automatically

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        npert : int
            The number of NCMC perturbation-propagation kernels to use.
        nprop : int
            The number of propagation steps per perturbation kernel
        mode : string
            Either 'add salt' or 'remove  salt', which is passed to 'updateForces'
        exchange_indices : numpy array
            Two element vector containing the residue indices that have been changed
        propagator : str
            The name of propagator.

        Returns
        -------
        work: float
            The work for appropriate for the stated propagator in units of KT.

        """
        self.integrator.setCurrentIntegrator(1)
        if propagator == 'velocityVerlet':
            vv = self.integrator.getIntegrator(1)
            # Get initial total energy
            logp_initial, pot, kin = self._compute_log_probability(context)
            # Propagation
            self.integrator.step(nprop)
            for stage in range(npert):
                # Perturbation
                self.updateForces(mode,exchange_indices,stage)
                self.forces_to_update.updateParametersInContext(context)
                # Propagation
                vv.step(nprop)
            # Get final total energy and calculate total work
            logp_final, pot, kin = self._compute_log_probability(context)
            work =  logp_initial - logp_final
        elif propagator == 'GHMC':
            ghmc = self.integrator.getIntegrator(1)
            ghmc.setGlobalVariableByName("ntrials", 0)      # Reset the internally accumulated work
            ghmc.setGlobalVariableByName("naccept", 0)
            # Propagation
            ghmc.step(1)
            for stage in range(npert):
                # Perturbation
                self.updateForces(mode,exchange_indices,stage)
                self.forces_to_update.updateParametersInContext(context)
                # Propagation
                ghmc.step(1)
            # Extract the internally calculated work from the integrator
            work = ghmc.getGlobalVariableByName('work') / self.kT_unitless
            self.naccepted_ghmc.append(ghmc.getGlobalVariableByName('naccept')/ghmc.getGlobalVariableByName('ntrials'))
        elif propagator == 'GHMC_old':
            # Like the GHMC integrator above, except that energies are calculated with getPotEnergy() for testing and benchmarking
            # Created due to errors in the energy calculations with CustomIntegrator.
            ghmc = self.integrator.getIntegrator(1)
            work = 0.0    # Unitless work
            # Propagation
            ghmc.step(nprop)
            for stage in range(npert):
                # Getting the potential energy before the perturbation
                pot_initial = self.getPotEnergy(context)
                # Perturbation
                self.updateForces(mode,exchange_indices,stage)
                self.forces_to_update.updateParametersInContext(context)
                # Getting the potential energy after the perturbation
                pot_final = self.getPotEnergy(context)
                # Propagation
                ghmc.step(nprop)
                # Update the accumulated work
                work += (pot_final - pot_initial)/self.kT
            self.naccepted_ghmc.append(ghmc.getGlobalVariableByName('naccept')/ghmc.getGlobalVariableByName('ntrials'))
        else:
            raise Exception('Propagator "{0}" not recognized'.format(propagator))
        self.integrator.setCurrentIntegrator(0)

        return work

    def setIdentity(self,mode,exchange_indices):
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


    def updateForces(self,mode,exchange_indices,stage=0):
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
            for atm1,atm2 in zip(molecule1,molecule2):
                self.forces_to_update.setParticleParameters(atm1.index,charge=self.wat2cat_parampath[atm_index]['charge'][stage],sigma=self.wat2cat_parampath[atm_index]['sigma'][stage],epsilon=self.wat2cat_parampath[atm_index]['epsilon'][stage])
                self.forces_to_update.setParticleParameters(atm2.index,charge=self.wat2an_parampath[atm_index]['charge'][stage],sigma=self.wat2an_parampath[atm_index]['sigma'][stage],epsilon=self.wat2an_parampath[atm_index]['epsilon'][stage])
                atm_index += 1
        if mode == 'remove salt':
            molecule1 = [atom for atom in self.mutable_residues[exchange_indices[0]].atoms()]
            molecule2 = [atom for atom in self.mutable_residues[exchange_indices[1]].atoms()]
            atm_index = 0
            for atm1,atm2 in zip(molecule1,molecule2):
                self.forces_to_update.setParticleParameters(atm1.index,charge=self.cat2wat_parampath[atm_index]['charge'][stage],sigma=self.cat2wat_parampath[atm_index]['sigma'][stage],epsilon=self.cat2wat_parampath[atm_index]['epsilon'][stage])
                self.forces_to_update.setParticleParameters(atm2.index,charge=self.an2wat_parampath[atm_index]['charge'][stage],sigma=self.an2wat_parampath[atm_index]['sigma'][stage],epsilon=self.an2wat_parampath[atm_index]['epsilon'][stage])
                atm_index += 1


    def updateForces_fractional(self,mode,exchange_indices,fraction=1.0):
        """
        Update the forcefield parameters accoring depending on whether inserting salt or water.

        Parameters
        ----------
        mode : string
            Whether the supplied indices will be used to 'add salt' or 'remove salt'
        exchange_indices : numpy array
            Which water residues will be converted to cation and anion, or which cation and anion will be turned
            into 2 water residue.
        fraction : float
            The fraction along the salt-water forcefield transformation pathway.

        """
        # Currently takes approx. 46 seconds per 100000 updates.

        if mode == 'add salt':
            initial_force = self.water_parameters
            # First, adding the cation.
            #print(mode,exchange_indices[0])
            molecule = [atom for atom in self.mutable_residues[exchange_indices[0]].atoms()]
            atm_index = 0
            for atom in molecule:
                target_force = self.cation_parameters[atm_index]
                charge = (1-fraction)*initial_force[atm_index]["charge"] + fraction*target_force["charge"]
                sigma = (1-fraction)*initial_force[atm_index]["sigma"] + fraction*target_force["sigma"]
                epsilon = (1-fraction)*initial_force[atm_index]["epsilon"] + fraction*target_force["epsilon"]
                self.forces_to_update.setParticleParameters(atom.index,charge=charge,sigma=sigma,epsilon=epsilon)
                atm_index += 1
            # Second, adding the anion.
            #print(mode,exchange_indices[1])
            molecule = [atom for atom in self.mutable_residues[exchange_indices[1]].atoms()]
            atm_index = 0
            for atom in molecule:
                target_force = self.anion_parameters[atm_index]
                charge = (1-fraction)*initial_force[atm_index]["charge"] + fraction*target_force["charge"]
                sigma = (1-fraction)*initial_force[atm_index]["sigma"] + fraction*target_force["sigma"]
                epsilon = (1-fraction)*initial_force[atm_index]["epsilon"] + fraction*target_force["epsilon"]
                self.forces_to_update.setParticleParameters(atom.index,charge=charge,sigma=sigma,epsilon=epsilon)
                atm_index += 1
        if mode == 'remove salt':
            #print(mode,exchange_indices[0])
            molecule = [atom for atom in self.mutable_residues[exchange_indices[0]].atoms()]
            initial_force = self.cation_parameters      # exchange_indices[0] is the cation residue.
            atm_index = 0
            for atom in molecule:
                target_force = self.water_parameters[atm_index]
                charge = (1-fraction)*initial_force[atm_index]["charge"] + fraction*target_force["charge"]
                sigma = (1-fraction)*initial_force[atm_index]["sigma"] + fraction*target_force["sigma"]
                epsilon = (1-fraction)*initial_force[atm_index]["epsilon"] + fraction*target_force["epsilon"]
                self.forces_to_update.setParticleParameters(atom.index,charge=charge,sigma=sigma,epsilon=epsilon)
                atm_index += 1
            #print(mode,exchange_indices[1])
            molecule = [atom for atom in self.mutable_residues[exchange_indices[1]].atoms()]
            initial_force = self.anion_parameters       # exchange_indices[1] is the anion residue.
            atm_index = 0
            for atom in molecule:
                target_force = self.water_parameters[atm_index]
                charge = (1-fraction)*initial_force[atm_index]["charge"] + fraction*target_force["charge"]
                sigma = (1-fraction)*initial_force[atm_index]["sigma"] + fraction*target_force["sigma"]
                epsilon = (1-fraction)*initial_force[atm_index]["epsilon"] + fraction*target_force["epsilon"]
                self.forces_to_update.setParticleParameters(atom.index,charge=charge,sigma=sigma,epsilon=epsilon)
                atm_index += 1

    def getPotEnergy(self,context):
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
        log_P = - total_energy/self.kT

        if self.pressure is not None:
            # Add pressure contribution for periodic simulations.
            volume = context.getState().getPeriodicBoxVolume()
            log_P += -self.pressure * volume * units.AVOGADRO_CONSTANT_NA/self.kT

        # Return the log probability.
        return log_P, pot_energy, kin_energy

    def update(self, context,nattempts=None,cost=None,saltmax=None):
        """
        Perform a number of Monte Carlo update trials for the titration state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        nattempts : integer
            Number of salt insertion and deletion moves to attempt.

        """
        if nattempts == None: nattempts = self.nattempts_per_update
        if cost == None:
            cost = [self.delta_chem/self.kT, -self.delta_chem/self.kT]      # [free energy to add salt, free energy to remove salt]
        elif type(cost) == units.Quantity:
            cost = [cost/self.kT, -cost/self.kT]
        elif type(cost) == float or type(cost) == int:
            cost = [cost, -cost]
        else:
            raise Exception('Penalty (delta_chem) for adding or removing salt "{0}" not recognized'.format(cost))
        # Perform a number of protonation state update trials.
        for attempt in range(nattempts):
            self.attempt_identity_swap(context,penalty=cost,saltmax=saltmax)
        return

    def getAcceptanceProbability(self):
        """
        Return the fraction of accepted moves

        Returns
        -------

        fraction : float
            the fraction of accepted moves

        """
        return float(self.naccepted) / float(self.nattempted)

    def getIdentityCounts(self):
        """
        Returns the total number of waters, cations, and anions

        Returns
        -------

        counts : tuple of integers
            The number of waters, cations, and anions respectively

        """
        nwats = np.sum(self.stateVector==0)
        ncation = np.sum(self.stateVector==1)
        nanion = np.sum(self.stateVector==2)
        return (nwats,ncation,nanion)
