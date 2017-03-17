import numpy as np
from saltswap.swapper import Swapper
import simtk.unit as unit
from copy import deepcopy

class Perturbator(Swapper):
    """
    Class to alchemical perturb the non-bonded parameters in a Swapper object in order to calculate energies and
    gradients with respect the to alchemical path. These values can be used to calculate relative hydration free
    energies using MBAR and TI. The gradients can also be used to estimate the thermodynamic length along the alchemical
    path.
    """
    def __init__(self, topology, system, integrator, context, mode, state, nstates=20, residues_indices=None,
                 temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres, waterName='HOH',
                 cationName='Na+', anionName='Cl-'):

        """
        Initialize the alchemical class to perturb the non bonded parameters of a Swapper object.

        Parameters
        ----------
        topology: simtk.openmm.topology
            topology of the system being simulated
        system: simtk.openmm.System
            openmm system being simulated
        integrator: simtk.openmm.Integrator
            openmm integrator that is used to propagate the dynamics
        context: simtk.openmm.context
            openmm context of the simulation system
        mode: str
            Whether to calculate the free energy to add salt or remove it. Either 'add salt' or 'remove salt'.
        state: int
            the index of the alchemical state that will be simulated. Indexing starts at 0 and ends at nstates - 1.
        nstates: int
            the total number of alchemical intermediates along the alchemical path
        residues_indices: list like object with 2 integers
            the indexes of the two residues that will be transformed either to or from salt. Two inth
        temperature: simtk.unit
            the temperature of simulation
        pressure: simtk.unit
            the simulation pressure
        waterName: str
            the name of the water molecule residues
        cationName: str
            the name of the cation residue
        anionName: str
            the name of anion residue
        """

        super(Perturbator, self).__init__(system=system, topology=topology, temperature=temperature, delta_chem=0.0,
                                          integrator=integrator, pressure=pressure, nattempts_per_update=0,
                                          npert=nstates - 1, nprop=0, propagator='GHMC', waterName=waterName,
                                          cationName=cationName, anionName=anionName)

        self.context = context
        self.nstates = nstates
        self.state = state

        mode_list = ('remove salt', 'add salt')
        if mode in mode_list:
            self.mode = mode
        else:
            raise Exception('The `mode` must be either {0}'.format(mode_list))

        if residues_indices is None:
            self.residues_indices = self.pick_residues()

    def pick_residues(self):
        """
        Randomly pick either pair of water molecules to turn into salt, or a cation-anion pair to turn into water.

        Returns
        -------
        exchange_indices: numpy array
            The residue numbers of the two molecules that will be turned into salt or water as specified by self.mode.
        """
        if self.mode == 'add salt':
            waters = np.where(self.stateVector == 0.0)[0]
            exchange_indices = np.random.choice(a=waters, size=2, replace=False)
        else:
            cations = np.random.choice(a=np.where(self.stateVector == 1)[0], size=1)
            anions = np.random.choice(a=np.where(self.stateVector == 2)[0], size=1)
            exchange_indices = np.array([cations, anions])
        return exchange_indices

    def change_stage(self, new_stage):
        """
        Update the parameters in the system to the ones specified by stage

        Parameters
        ----------
        new_stage: int
            The index of the parameter path to which parameters will be updated to

        """
        self.state = new_stage
        self._update_forces(self.mode, self.residues_indices, stage=new_stage)
        self.forces_to_update.updateParametersInContext(self.context)

    def perturb_energy(self, perturbed_stage, in_thermal_units=True):
        """
        Calculate the energy of the current configuration at a different value of the non-bonded parameters for the
        pair of water or ions. The possible values of the non-bonded parameters are preset and discretized into
        different stages, so that the perturbation is to the parameters at the different stage to the current one.

        Parameters
        ----------
        perturbed_stage: int
            the stage along the perturbation (lambda) path where the energy will be calculated
        in_thermal_units: bool
            whether to return the energy in thermal units, transforming the energies to float

        Returns
        -------
        energy: simtk.unit or float
            the energy of the configuration at the parameters in the perturbed stage. Only float if
            in_thermal_units=True.
        """

        # Perturb
        self._update_forces(self.mode, self.residues_indices, perturbed_stage)
        self.forces_to_update.updateParametersInContext(self.context)

        # Calculate energy
        logp, pot_energy, kin_energy = self._compute_log_probability(self.context)
        total_energy = pot_energy + kin_energy
        if in_thermal_units:
            total_energy = total_energy / self.kT

        # Return to initial state
        self._update_forces(self.mode, self.residues_indices, stage=self.state)
        self.forces_to_update.updateParametersInContext(self.context)

        return total_energy

    def perturb_all_states(self, in_thermal_units=True):
        """
        Calculate the energy of the current configuration at the non-bonded parameter values at each stage

        Parameters
        ----------
        in_thermal_units: bool
            whether to return the energy in thermal units, transforming the energies to float

        Returns
        -------
        energies: list
            A list of all the configuration energies at every stage along the parameter path
        """
        return [self.perturb_energy(stage, in_thermal_units) for stage in range(self.nstates)]

    def _update_force_along_path(self, param_path, residue_index, dlambda=0.001):
        """
        Perturb the non-bonded parameters by a given fraction along the discrete path specified by param_path.

        Parameters
        ----------
        param_path: numpy array
            the list of the dictionary of parameters along the alcehmical path
        residue_index: int
            the index of the residue whose parameters will be perturbed along the path
        dlambda: float
            the fraction along the path along which the parameter will be linearly extrapolated.
        """
        if self.state != self.nstates - 1:
            along = 1
        else:
            along = -1
        # The fraction along the parameter path adjacent stages are, assuming the stages are spaced linearly along the
        # path.
        delta_path = float(along) / float(self.nstates)

        molecule = [atom for atom in self.mutable_residues[residue_index].atoms()]
        atm_index = 0
        for atom in molecule:
            param_gradient = (param_path[atm_index]["charge"][self.state + along] - param_path[atm_index]["charge"][self.state]) / delta_path
            charge = param_path[atm_index]["charge"][self.state] + dlambda * param_gradient

            param_gradient = (param_path[atm_index]["sigma"][self.state + along] - param_path[atm_index]["sigma"][self.state]) / delta_path
            sigma = param_path[atm_index]["sigma"][self.state] + dlambda * param_gradient

            param_gradient = (param_path[atm_index]["epsilon"][self.state + along] - param_path[atm_index]["epsilon"][self.state]) / delta_path
            epsilon = param_path[atm_index]["epsilon"][self.state] + dlambda * param_gradient

            self.forces_to_update.setParticleParameters(atom.index, charge=charge, sigma=sigma, epsilon=epsilon)
            atm_index += 1

    def estimate_energy_gradient(self, dlambda=0.001):
        """
        Numerically estimate the gradient of the energy along the pre-defined parameter path by the finite difference
        approximation. If the perturbation path is parameterized by l, then this function estimates

            dU/dl = (U(l + delta_l) - U(l)) / delta_l

        where U(l + delta_l) is approximated using the finite difference approximation in parameter space in with the
        help of _update_force_along_path.

        Parameter
        ---------
        dlambda: float
            the difference along the perturbation path that will be used in the finite difference approximation.
        """

        energy = self._get_potential_energy(self.context)

        if self.mode == 'add salt':
            self._update_force_along_path(self.wat2cat_parampath, self.residues_indices[0], dlambda)
            self._update_force_along_path(self.wat2an_parampath, self.residues_indices[1], dlambda)
        else:
            self._update_force_along_path(self.cat2wat_parampath, self.residues_indices[0], dlambda)
            self._update_force_along_path(self.an2wat_parampath, self.residues_indices[1], dlambda)

        self.forces_to_update.updateParametersInContext(self.context)
        new_energy = self._get_potential_energy(self.context)

        gradient = (new_energy - energy) / dlambda

        # Return to initial state
        self._update_forces(self.mode, self.residues_indices, stage=self.state)
        self.forces_to_update.updateParametersInContext(self.context)

        return gradient

    def gradients_all_stages(self, dlambda=0.001, in_thermal_units=True):
        """
        Estimate the gradient with respect to alchemical path at each state given the current configuration

        Parameter
        ---------
        dlambda: float
            the fraction along the alchemical path that's used in the finite difference approximation to the gradient

        Returns
        -------
        gradients: numpy array
            array of the estimate gradients at each alchemical state
        """

        original_state = deepcopy(self.state)

        if in_thermal_units:
            denominator = self.kT
        else:
            denominator = 1.0

        # Perturbing the alchemical states and calculating the gradient
        gradients = []
        for stage in range(self.nstates):
            self.change_stage(stage)
            gradients.append(self.estimate_energy_gradient(dlambda) / denominator)
        # Return back to the current state
        self.change_stage(original_state)

        return gradients