from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app
import numpy as np
import copy
from mcmc_samplers import MCMCSampler
from saltswap import strip_in_unit_system
import pytest

class TestParamPerturbations(object):
    """
    Class to verify the correct handling of non-bonded parameters in SaltSwap before and after exchange moves.
    """

    def _get_params(self, saltswap, mol_ind):
        """
        Retrieve the non-bonded parameters of a particular molecule within a saltswap object.

        Parameters
        ----------
        saltswap : saltswap object
          Saltswap driver that is manipulating the non-bonded parameters of the molecules of interest
        mol_ind : int
          index of the molecule

        Return
        ------
        param_matrix : numpy array
          matrix whose rows contain the charge, sigma, and epilson parameters of each atom in the molecule
        """
        molecule = [atom for atom in saltswap.mutable_residues[mol_ind].atoms()]
        param_matrix = np.zeros((len(molecule),3))
        param_index = 0
        for atom in molecule:
            (charge, sigma, epsilon) = saltswap.forces_to_update.getParticleParameters(atom.index)
            param_matrix[param_index,0] = strip_in_unit_system(charge)
            param_matrix[param_index,1] = strip_in_unit_system(sigma)
            param_matrix[param_index,2] = strip_in_unit_system(epsilon)
            param_index += 1
        return param_matrix

    def test_param_finder(self):
        """
        Saltswap extracts the non-bonded paramaters of TIP3P water from the system. This function confirms this is done
        correctly using a box of water.
        """
        size = 15.0 * unit.angstrom
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        state = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = 0, nprop = 0, npert = 1)
        saltswap = state.saltswap

        # Extract the water parameters of the system
        param_matrix = self._get_params(saltswap,0)

        # Make sure they match with saltswaps parameters for water
        param_names = ('charge', 'sigma', 'epsilon')
        for i in range(len(saltswap.water_parameters)):
            for j in range(len(param_names)):
                assert param_matrix[i,j] == saltswap.water_parameters[i][param_names[j]]

    def test_salt_insertions_instant(self):
        """
        Verifies whether the saltswap non-bonded parameter exchanges do not affect the end states for instantaneous
        moves. This function tests the parameters in the perturbations water --> cation and water --> anion.

        The test system is a box of water.
        """

        Dmu_insert = -10000.0
        size = 15.0 * unit.angstrom
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        state = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = Dmu_insert, nprop = 0, npert = 1)
        saltswap = state.saltswap

        # Get the parameters of water before any perturbations
        water_inital_params = copy.deepcopy( self._get_params(saltswap,0) )

        # Insert salt using a chemical potential that makes this highly likely. Stop when 1 salt pair has been inserted.
        nosalt = True
        while nosalt :
            saltswap.update(state.context, nattempts = 1, cost = Dmu_insert)
            n_wats ,n_ions, n_ions = saltswap.getIdentityCounts()
            nosalt = (n_ions == 0)

        # Get the parameters of the anion and cations
        cation_index = np.where(saltswap.stateVector == 1)[0][0]
        cation_matrix = self._get_params(saltswap, cation_index)
        anion_index = np.where(saltswap.stateVector == 2)[0][0]
        anion_matrix = self._get_params(saltswap, anion_index)

        # Compare the parameters with their target values
        param_names = ('charge', 'sigma', 'epsilon')
        for i in range(len(saltswap.water_parameters)):
            for j in range(len(param_names)):
                assert cation_matrix[i,j] == saltswap.cation_parameters[i][param_names[j]]
                assert anion_matrix[i,j] == saltswap.anion_parameters[i][param_names[j]]

    def test_salt_insertions_ncmc(self):
        """
        Verifies whether the saltswap non-bonded parameter exchanges do not affect the end states for NCMC
        switches. This function tests the parameters in the perturbations water --> cation and water --> anion.

        The test system is a box of water.
        """

        Dmu_insert = -10000.0
        size = 15.0 * unit.angstrom
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        state = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = Dmu_insert, nprop = 1, npert = 10)
        saltswap = state.saltswap

        # Insert salt using a chemical potential that makes this highly likely. Stop when 1 salt pair has been inserted.
        nosalt = True
        while nosalt :
            saltswap.update(state.context, nattempts = 1, cost = Dmu_insert)
            n_wats ,n_ions, n_ions = saltswap.getIdentityCounts()
            nosalt = (n_ions == 0)

        # Get the parameters of the anion and cations
        cation_index = np.where(saltswap.stateVector == 1)[0][0]
        cation_matrix = self._get_params(saltswap, cation_index)
        anion_index = np.where(saltswap.stateVector == 2)[0][0]
        anion_matrix = self._get_params(saltswap, anion_index)

        # Compare the parameters with their target values
        param_names = ('charge', 'sigma', 'epsilon')
        for i in range(len(saltswap.water_parameters)):
            for j in range(len(param_names)):
                assert cation_matrix[i,j] == saltswap.cation_parameters[i][param_names[j]]
                assert anion_matrix[i,j] == saltswap.anion_parameters[i][param_names[j]]

    def test_salt_deletions_instant(self):
        """
        Confirms the correct end state non-bonded parameters for the cyclic perturbations water --> cation --> water
        and water --> anion --> water. Testing instantaneous switches in a box of water.
        """

        Dmu_insert = -10000.0
        size = 15.0 * unit.angstrom
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        state = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = Dmu_insert, nprop = 0, npert = 1)
        saltswap = state.saltswap

        # Get the parameters of water before any perturbations
        water_inital_params = copy.deepcopy( self._get_params(saltswap,0) )

        # Insert salt using a chemical potential that makes this highly likely. Stop when 1 salt pair has been inserted.
        nosalt = True
        while nosalt :
            saltswap.update(state.context, nattempts = 1, cost = Dmu_insert)
            n_wats ,n_ions, n_ions = saltswap.getIdentityCounts()
            nosalt = (n_ions == 0)

        # Get the parameters of the new anion and cations that were previously water
        cation_index = np.where(saltswap.stateVector == 1)[0][0]
        anion_index = np.where(saltswap.stateVector == 2)[0][0]

        # Now delete the anion and cation, using the negative of the original chemical potential
        while not nosalt :
            saltswap.update(state.context, nattempts = 1, cost = -Dmu_insert)
            n_wats ,n_ions, n_ions = saltswap.getIdentityCounts()
            nosalt = (n_ions == 0)

        # Check that parameters of the deleted anion and cation match the orginal water parameters
        assert np.sum (water_inital_params - self._get_params(saltswap, cation_index)) == 0.0
        assert np.sum (water_inital_params - self._get_params(saltswap, anion_index)) == 0.0

    def test_salt_deletions_ncmc(self):
        """
        Confirms the correct end state non-bonded parameters for the cyclic perturbations water --> cation --> water
        and water --> anion --> water. Testing NCMC switches in a box of water.
        """
        Dmu_insert = -10000.0
        size = 15.0 * unit.angstrom
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        state = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = Dmu_insert, nprop = 1, npert = 10)
        saltswap = state.saltswap

        # Get the parameters of water before any perturbations
        water_inital_params = copy.deepcopy( self._get_params(saltswap,0) )

        # Insert salt using a chemical potential that makes this highly likely. Stop when 1 salt pair has been inserted.
        nosalt = True
        while nosalt :
            saltswap.update(state.context, nattempts = 1, cost = Dmu_insert)
            n_wats ,n_ions, n_ions = saltswap.getIdentityCounts()
            nosalt = (n_ions == 0)

        # Get the parameters of the new anion and cations that were previously water
        cation_index = np.where(saltswap.stateVector == 1)[0][0]
        anion_index = np.where(saltswap.stateVector == 2)[0][0]

        # Now delete the anion and cation, using the negative of the original chemical potential
        while not nosalt :
            saltswap.update(state.context, nattempts = 1, cost = -Dmu_insert)
            n_wats ,n_ions, n_ions = saltswap.getIdentityCounts()
            nosalt = (n_ions == 0)

        # Check that parameters of the deleted anion and cation match the orginal water parameters
        assert np.sum (water_inital_params - self._get_params(saltswap, cation_index)) == 0.0
        assert np.sum (water_inital_params - self._get_params(saltswap, anion_index)) == 0.0