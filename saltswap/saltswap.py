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

class SaltSwap(object):
    """
    Monte Carlo driver for semi-grand canonical ensemble moves.

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
        self.pressure = pressure
        self.delta_chem = delta_chem
        self.debug = debug

        self.anionName = anionName
        self.cationName = cationName
        self.waterName = waterName

        self.integrator = integrator

        proplist = ['GHMC','velocityVerlet']
        if propagator in ['GHMC','velocityVerlet']:
            self.propagator = propagator
        elif propagator not in ['GHMC','velocityVerlet'] and npert==0:
            pass
        else:
            raise Exception('NCMC propagator {0} not in supported list {1}'.format(propagator,proplist))

        self.npert  = npert
        self.nprop = nprop

        # Set constraint tolerance.
        #self.verlet_integrator.setConstraintTolerance(integrator.getConstraintTolerance())

        # Store force object pointer.
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if force.__class__.__name__ == 'NonbondedForce':
                self.forces_to_update = force

        # Check that system has MonteCarloBarostat if pressure is specified.
        if pressure is not None:
            forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
            if 'MonteCarloBarostat' not in forces:
                raise Exception("`pressure` is specified, but `system` object lacks a `MonteCarloBarostat`")
                self.barofreq = None
            else:
                self.barostat = forces['MonteCarloBarostat']
                self.barofreq = self.barostat.getFrequency()

        self.mutable_residues = self.identifyResidues(self.topology,residue_names=(self.waterName,self.anionName,self.cationName))

        self.stateVector = self.initializeStateVector()
        self.water_parameters = self.retrieveResidueParameters(self.topology,self.waterName)
        self.cation_parameters = self.initializeIonParameters(ion_name=self.cationName,ion_params=None)
        self.anion_parameters = self.initializeIonParameters(ion_name=self.anionName,ion_params=None)

        # Setting the perturbation pathway for NCMC
        self.set_parampath()



        # Describing the identities of water and ions with numpy vectors

        # Track simulation state
        #self.kin_energies = units.Quantity(list(), units.kilocalorie_per_mole)
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

        # For counting the number of NaNs I get in NCMC. These are automatically rejected.
        self.nan = 0
        return

    def set_parampath(self):

        # DONE: initialzie arrays to store param paths
        # DONE: check params with old function
        # TODO: Ensure correct sampling with null swap test script
        # TODO: After checks, no need to calculate and save parameters outside this function. Make funcitons below private
        # DONE: Change if statement in attempt_identity_swap so that instantaneous swaps only occur for nprop=0

        #wat_params = self.retrieveResidueParameters(self.topology,self.waterName)
        #cat_params = self.initializeIonParameters(ion_name=self.cationName,ion_params=None)
        #an_params = self.initializeIonParameters(ion_name=self.anionName,ion_params=None)
        wat_params = self.water_parameters
        cat_params = self.cation_parameters
        an_params = self.anion_parameters

        # Pre-assigment of the data structures to store the perturbation path
        self.wat2cat_parampath = []
        self.wat2an_parampath = []
        self.cat2wat_parampath = []
        self.an2wat_parampath = []
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
                    parameters = {'charge': charge, 'sigma': sigma, 'epsilon': epsilon}
                    param_list.append(parameters)
                    #if self.debug: print('retrieveResidueParameters: %s : %s' % (resname, str(parameters)))
                return param_list
        raise Exception("resname '%s' not found in topology" % resname)

    def initializeIonParameters(self,ion_name,ion_params=None):
        '''
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
        '''

        # Creating a list of non-bonded parameters that matches the size of the water model.
        num_wat_atoms = len(self.water_parameters)

        # TODO: set eps to exactly zero to verify bug
        # Initialising dummy atoms to having the smallest float that's not zero, due to a bug
        eps = sys.float_info.epsilon
        ion_param_list = num_wat_atoms*[{'charge': eps*units.elementary_charge,'sigma': eps*units.nanometer,'epsilon':eps*units.kilojoule_per_mole}]

        # Making the first element of list of parameter dictionaries the ion. This means that ions will be centered
        # on the water oxygen atoms.
        # If ion parameters are not supplied, use Joung and Cheatham parameters.
        if ion_name == self.cationName:
            if ion_params == None:
                ion_param_list[0] = {'charge': 1.0*units.elementary_charge, 'sigma': 0.2439281*units.nanometer, 'epsilon': 0.0874393*units.kilocalorie_per_mole}
            else:
                ion_param_list[0] = ion_params
        elif ion_name == self.anionName:
            if ion_params == None:
                ion_param_list[0] = {'charge': -1.0*units.elementary_charge,'sigma': 0.4477657*units.nanometer,'epsilon':0.0355910*units.kilocalorie_per_mole}
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

        TODO
        ----
        * Can this feature be added to simt.openmm.app.Topology?

        """
        target_residues = list()
        for residue in topology.residues():
            if residue.name in residue_names:
                target_residues.append(residue)

        if self.debug: print('identifyResidues: %d %s molecules identified.' % (len(target_residues),residue_names[0]))
        return target_residues

    def initializeStateVector(self):
        '''
        Stores the identity of the mutabable residues in a numpy array for efficient seaching and updating of
        residue identies.

        Returns
        -------
        stateVector : numpy array
            Array of 0s, 1s, and 2s to indicate water, sodium, and chlorine.

        '''
        names = [res.name for res in self.mutable_residues]
        stateVector = np.zeros(len(names))
        for i in range(len(names)):
            if names[i] == self.waterName:  stateVector[i] = 0
            elif names[i] == self.cationName: stateVector[i] = 1
            elif names[i] == self.anionName: stateVector[i] = 2
        return stateVector


    def resetStatistics(self):
        """
        Reset statistics of titration state tracking.

        Todo
        ----

        * Keep track of more statistics regarding history of individual protonation states.
        * Keep track of work values for individual trials to use for calibration.

        """

        self.nattempted = 0
        self.naccepted = 0

        return

    def attempt_identity_swap(self,context,penalty,saltmax=None):
        '''
        Attempt the exchange of (possibly multiple) chemical species.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        penalty : list floats
            The free energy to add salt (first index) and remove salt (second index)
        saltmax : float
            The maximum number of salt pairs that you wish to be added. If None, then the maximum number is the
            number of water molecules divided by 2.

        Notes
        -----
        Code currently written specifically for exchanging two water molecules for Na and Cl, with generalisation to follow.
        '''
        self.nattempted += 1

        if type(penalty)==float:
            penalty = [penalty,-penalty]

        if self.barostat is not None:
            self.barostat.setFrequency(0)

        # If using NCMC, store initial positions.
        if self.nprop > 0:
            initial_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
            initial_velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)

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

        # Compute initial energy
        #logP_initial, pot1, kin1 = self._compute_log_probability(context)

        # Perform perturbation to remove or add salt with NCMC and calculate energies
        if self.nprop > 0:
            try:
                work = self.NCMC(context,self.npert,self.nprop,mode_forward,change_indices,propagator=self.propagator)
            except Exception as detail:
                work = 1000000000000.0               # If the simulation explodes during NCMC, reject with high work
                if detail[0]=='Particle coordinate is nan': self.nan += 1
            #logP_final, pot2, kin2 = self._compute_log_probability(context)
        else:
            # OLD WAY
            #logP_initial, pot1, kin1 = self._compute_log_probability(context)
            #self.updateForces_fractional(mode_forward,change_indices,fraction=1.0)
            #self.forces_to_update.updateParametersInContext(context)
            #logP_final, pot2, kin2 = self._compute_log_probability(context)
            #work = logP_initial - logP_final
            #print('OLD',logP_final, type(logP_final))
            #######################
            # NEW WAY
            #logP_initial, pot1, kin1 = self._compute_log_probability(context)
            pot_initial = self.getPotEnergy(context)
            self.updateForces(mode_forward,change_indices,stage=0)
            self.forces_to_update.updateParametersInContext(context)
            pot_final= self.getPotEnergy(context)
            work = (pot_final - pot_initial)/self.kT
            #logP_final, pot2, kin2 = self._compute_log_probability(context)
            #print('NEW',logP_final,type(logP_final))
            #work = logP_initial - logP_final


        # Computing the work after velocity Verlet: Work = E_final - E_initial
        #work = logP_initial - logP_final
        if mode_forward == "remove salt":
            self.work_rm.append(work)
        else:
            self.work_add.append(work)

        # Cost = F_final - F_initial, where F_initial is the free energy to have the current number of salt molecules.
        # log_accept += cost - work
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
            self.updateForces(mode_backward,change_indices,stage=0)
            #self.updateForces_fractional(mode_backward,change_indices,fraction=1.0)
            self.forces_to_update.updateParametersInContext(context)
            if self.nprop > 0:
                context.setPositions(initial_positions)
                context.setVelocities(initial_velocities)

        if self.barostat is not None:
            self.barostat.setFrequency(self.barofreq)


    def NCMC(self,context,npert,nprop,mode,exchange_indices,propagator='GHMC'):
        """
        Performs nonequilibrium candidate Monte Carlo for the addition or removal of salt.
        So that the protocol is time symmetric, the protocol is given by
             propagation -> perturbation -> propagation


        TODO: have the propagation kernel type read automatically

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        npert : integer
            The number of NCMC perturbation-propagation kernels to use.
        nsteps : integer
            The number of velocity verlet steps to take in the propagation kernel
        mode : string
            Either 'add salt' or 'remove  salt'
        exchange_indices : numpy array
            Two element vector containing the residue indices that have been changed

        Returns
        -------
        work: simtk.unit
            The work for appropriate for the stated propagator

        """
        self.integrator.setCurrentIntegrator(1)
        if propagator == 'velocityVerlet':
            # Get initial energy
            pot_initial = self.getPotEnergy(context)
            # Propagation
            #print('preloop pot_old=',pot_initial)
            self.integrator.step(nprop)
            #print('preloop pot_new=',self.getPotEnergy(context))
            for stage in range(npert):
                # Perturbation
                # OLD way
                #fraction = float(stage + 1)/float(npert)
                #self.updateForces_fractional(mode,exchange_indices,fraction)
                #self.forces_to_update.updateParametersInContext(context)
                # New way
                self.updateForces(mode,exchange_indices,stage)
                self.forces_to_update.updateParametersInContext(context)
                # Propagation
                self.integrator.step(nprop)
            # Get final energy and calculate total work
            pot_final = self.getPotEnergy(context)
            work =  (pot_final -  pot_initial)/self.kT
        elif propagator == 'GHMC':
            work = 0    # Unitless work
            self.integrator.step(nprop)
            for stage in range(npert):
                pot_initial = self.getPotEnergy(context)
                # Perturbation
                self.updateForces(mode,exchange_indices,stage)
                #########################################
                # The old, slow way
                #fraction = float(stage + 1)/float(npert)
                #self.updateForces_fractional(mode,exchange_indices,fraction)
                #########################################
                self.forces_to_update.updateParametersInContext(context)
                # Update the accumulated work
                pot_final = self.getPotEnergy(context)
                #if debug == True:
                #    self.updateForces(mode,exchange_indices,stage)
                #    self.forces_to_update.updateParametersInContext(context)
                #    test_energy = self.getPotEnergy(context)
                #    print(pot_final,test_energy)
                work += (pot_final - pot_initial)/self.kT
                # Propagation
                self.integrator.step(nprop)
        else:
            raise Exception('Propagator "{0}" not recognized'.format(propagator))

        self.integrator.setCurrentIntegrator(0)

        return work

    def setIdentity(self,mode,exchange_indices):
        '''
        Function to set the names of the mutated residues and update the state vector. Called after a transformation
        of the forcefield parameters has been accepted.

        Parameters
        ----------
        mode : string
            Either 'add salt' or 'remove  salt'
        exchange_indices : numpy array
            Two element vector containing the residue indices that have been changed

        '''

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
        '''
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
        Returns
        -------

        '''
        # Initialise self.cation_parampath[atm_index][1,stage] and self.anion_parampath[atm_index][1,stage]
        #        list with 3 elements, each element contains a matrix with 3 rows for each parameter, and columns for the value of each parameter at a given NCMC stage


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
        '''
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
        Returns
        -------

        '''
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
        '''
        Extract the potential energy of the system

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to get the energy of
        Returns
        -------
        potential energy : qunatity in default unit of energy

        '''
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

        TODO
        ----
        * Generalize this to use ThermodynamicState concept of reduced potential (from repex)


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

        Notes
        -----
        The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.

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
        '''
        Returns the total number of waters, cations, and anions
        Returns
        -------

        counts : tuple of integers
            The number of waters, cations, and anions respectively

        '''
        nwats = np.sum(self.stateVector==0)
        ncation = np.sum(self.stateVector==1)
        nanion = np.sum(self.stateVector==2)
        return (nwats,ncation,nanion)
