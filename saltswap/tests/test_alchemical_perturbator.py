import numpy as np
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from perturbator import Perturbator

class TestPerturbator(object):
    """
    Class to test the functionality of the class that can calculate energies and gradients at different parameter values.
    """

    def create_example(self, stage=0, nstages=10, temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres):
        """
        Creates a box of water as a perturbation test system. This function simplifies the writing of the other tests.

        Parameters
        ----------
        stage: int
            the index of lambda value that the system is at
        nstages: int
            the total number of lambda values
        temperature: simtk.unit
            the temperature of the system
        pressure: simtk.unit
            the pressure of the system

        Returns
        -------

        perturber: saltswap.Perturbator
            class to perturb the non-bonded parameters of water and salt to each other.
        """

        # Create a box of water
        wbox = WaterBox(box_edge=25.0 * unit.angstrom , nonbondedMethod=app.PME)
        integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        wbox.system.addForce(barostat)
        context = openmm.Context(wbox.system, integrator)
        context.setPositions(wbox.positions)

        perturber = Perturbator(topology=wbox.topology, system=wbox.system, integrator=integrator, context=context,
                                mode='add salt', stage=stage, nstages=nstages, temperature=temperature, pressure=pressure)

        return perturber

    def test_initialization(self):
        """
        Test whether the initialization works as expected for a box of water
        """
        self.create_example()

    def test_perturb_energy(self):
        """
        Ensure energies are calculated correctly are output in the correct data structure.
        """
        s = 0
        perturber = self.create_example(stage=s)
        energy_float = perturber.perturb_energy(perturbed_stage=s + 1, in_thermal_units=True)
        energy_unit = perturber.perturb_energy(perturbed_stage=s + 1, in_thermal_units=False)

        assert (type(energy_float) == float) and (type(energy_unit) == unit.quantity.Quantity)


    def test_all_perturbations(self):
        """
        Test the basic functioning of the function used to calculate energies for the parameter values at all stages.
        """
        nstages = 10
        perturber = self.create_example(nstages=nstages)
        energies = perturber.perturb_all_states()

        assert len(energies) == nstages

    def test_grad_estimator(self):
        """
        Ensures the function to calculates the gradient at a given stage does not throw an error
        """
        perturber = self.create_example()
        perturber.estimate_energy_gradient()

    def test_grad_values(self):
        """
        Makes sure that the estimates for the gradients are approximately correct by comparing them to the differences
        in perturbation energies. As the number of states increases the two methods should converge to the same values.
        """
        nstages = 300
        perturber = self.create_example(nstages=nstages)

        # Estimate the gradient at each stage along the path in thermal units
        gradients = np.zeros(nstages)
        for stage in range(nstages):
            perturber.change_stage(stage)
            gradients[stage] = perturber.estimate_energy_gradient(dlambda=0.001) / perturber.kT
        gradients = np.array(gradients)

        # Now estimating the gradient by the difference in the perturbed energies
        energies = perturber.perturb_all_states(in_thermal_units=True)
        gradients_by_energy = perturber.nstages * np.diff(energies)

        fractional_error = np.mean(np.absolute(gradients_by_energy - gradients[:-1])) / np.mean(gradients_by_energy)

        # This is a stochastic test, so having a lenient maximum error of 30%.
        assert fractional_error < 0.3