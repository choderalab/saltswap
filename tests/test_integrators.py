from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
import copy
from integrators import GHMCIntegrator as GHMC
import pytest

def detect_cuda():
    """
    Detect whether CUDA is available to openmm on the platform where the test is being performed.
    This is used to determine whether to perform tests on platforms with CUDA.

    Return
    ------
    hasCUDA : bool
      whether CUDA is accessible to openmm on the platform

    """
    try:
        openmm.Platform.getPlatformByName('CUDA')
        noCUDA = False
    except Exception:
        noCUDA = True

    return noCUDA

def detect_opencl():
    """
    Detect whether CUDA is available to openmm on the platform where the test is being performed.
    This is used to determine whether to perform tests on platforms with OpenCL.

    Return
    ------
    hasCUDA : bool
      whether CUDA is accessible to openmm on the platform

    """
    try:
        openmm.Platform.getPlatformByName('OpenCL')
        noOpenCL = False
    except Exception:
        noOpenCL = True

    return noOpenCL

class TestIntegrators():
    """
    Tests the NCMC integrators in SaltSwap, using a box of water as the test system. In order to exclusively test the
    integrators, this class is a very simplified version of SaltSwap, and it defines its own functions for an
    NCMC procedure.

    Some of the tests are platform specific.
    """

    def _get_nonbonded_force(self, system):
        """
        Extracts the non-bonded force from a system for use in NCMC
        """
        forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
        return forces['NonbondedForce']

    def _update_forces(self, stage, npert, force, reference_force):
        """
        Fractionally reduces the charge of the first water molecule in the box by a fraction stage/npert.

        Parameters
        ----------
        stage : int
          The stage along the NCMC procedure
        reference_force : OpenMM force object
          Pointer to the non-bonded force of the system prior to any perturbations
        force : OpenMM force object
          The pointer to the non-bonded force of the system simulated
        """
        for index in range(3):
            [charge, sigma, epsilon] = reference_force.getParticleParameters(index)
            charge *= (1.0 - float(stage+1)) / float(npert)
            force.setParticleParameters(index, charge, sigma, epsilon)

    def _get_constants(self, temperature):
        """
        Returns Boltzmann's constant multiplied by temperature in units of kJ/K.

        Parameters
        ----------
        temperature : Qunatity in units of Kelvin

        Returns
        -------
        kT : Quantity in units of kJ/K
          Thermal energy
        kT_unitless : float
          Thermal energy in units of kJ/K
        """
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kB = kB.in_units_of(unit.kilojoule_per_mole / unit.kelvin)
        kT = kB * temperature
        kT_unitless = kT.value_in_unit_system(unit.md_unit_system)

        return kT, kT_unitless

    def _make_ghmc_system(self, nprop, temperature, collision_rate = 1.0 / unit.picoseconds, timestep = 1.0 * unit.femtoseconds):
        """
        Creates the water box test system with a langevin and ghmc compound integrator.

        Parameters
        ----------
        nprop: int
          The number of NCMC propagation steps per propagation step
        temperature: Quantity Kelvin
          The temperature at which the simulation will be performed
        collision_rate: Quantity 1/time
          Frequency of collisions for Langevin and GHMC integrators
        timestep: Quantity time
          the timestep for both integrators

        Returns
        -------
        wbox: openmmtools.testsystems object
          The box of water
        system: openmm system
          The system of the box of water
        reference_force: openmm force object
          the non-bonded force of the unperturbed system
        force : openmm force object
          the non-bonded force of the used in the simulation
        integrator : openmm.CompoundIntegrator
          the compound integrator
        ghmc : openmm.CustomIntegrator
          The GHMC integrator used in NCMC moves in SaltSwap
        langevin : openmm.integrator
          The Langevin integrator for non-NCMC moves.
        """

        # Make the water box test system
        wbox = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic)
        reference_system = wbox.system
        system = copy.deepcopy(wbox.system)
        reference_force = self._get_nonbonded_force(reference_system)
        force = self._get_nonbonded_force(system)

        # Create the compound integrator with SaltSwap's custom GHMC integrator
        ghmc = GHMC(temperature, collision_rate, timestep, nprop)
        langevin = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        integrator = openmm.CompoundIntegrator()
        integrator.addIntegrator(langevin)
        integrator.addIntegrator(ghmc)

        return wbox, system, reference_force, force, integrator, ghmc, langevin

    def test_ghmc_integrator_cpu(self):
        """
        Tests the GHMC integrator with an NCMC procedure on a box of water. The protocol work is calculated inside the
        integrator and externally with getState(getEnergy=True). The work calculated with both methods should agree.

        A compound integrator is used to make this test more like the simulation procedure with saltswap.

        This version runs the test on CUDA
        """
        temperature = 298.0 * unit.kelvin

        # NCMC parameters
        npert = 128          # Number of perturbation steps
        nprop = 2           # Number of propagation steps per perturbation

        # Get thermal energy
        (kT, kT_unitless) = self._get_constants(temperature)

        # Make the water box test system and return everything
        (wbox, system, reference_force, force, integrator, ghmc, langevin) = self._make_ghmc_system(nprop, temperature)

        # Create the context
        context = openmm.Context(system, integrator)
        context.setPositions(wbox.positions)

        # Take a few steps of langevin dynamics to test the compound integrator
        langevin.step(50)

        #### NCMC ####
        # Accumalating the work performed 3 ways.
        ext_work_integrator = 0.0    # Externally-accumulated unitless work calculated using integrator variables
        ext_work_getenergy = 0.0     # Externally-accumulated unitless work calculated using getEnergy()
        # The third work term is accumulated inside the integrator.

        # Steps are taken with GHMC, again making use of the compound integrator
        ghmc.step(1)
        for stage in range(npert):
            # Initial energy with getEnergy
            initial_energy_getenergy = context.getState(getEnergy=True).getPotentialEnergy() / kT
            # Initial energy from integrator
            initial_energy_integrator = ghmc.getGlobalVariableByName('potential_new') / kT_unitless
            # Perturbation
            self._update_forces(stage, npert, force, reference_force)
            force.updateParametersInContext(context)
            # Accumulate work (a)
            final_energy_getenergy = context.getState(getEnergy=True).getPotentialEnergy() / kT
            ext_work_getenergy += (final_energy_getenergy - initial_energy_getenergy)
            # Propagation
            ghmc.step(1)
            # Accumulate work (b)
            final_energy_integrator = ghmc.getGlobalVariableByName('potential_initial') / kT_unitless
            ext_work_integrator += (final_energy_integrator - initial_energy_integrator)

        # Both work estimates should agree.
        assert ( abs(ext_work_getenergy - ext_work_integrator) < 0.01 )

    @pytest.mark.skipif(detect_cuda(), reason="CUDA not detected on platform")
    def test_ghmc_integrator_cuda(self):
        """
        Tests the GHMC integrator with an NCMC procedure on a box of water. The protocol work is calculated inside the
        integrator and externally with getState(getEnergy=True). The work calculated with both methods should agree.

        A compound integrator is used to make this test more like the simulation procedure with saltswap.

        This version runs the test on CUDA
        """
        temperature = 298.0 * unit.kelvin

        # NCMC parameters
        npert = 128          # Number of perturbation steps
        nprop = 2           # Number of propagation steps per perturbation

        # Get thermal energy
        (kT, kT_unitless) = self._get_constants(temperature)

        # Make the water box test system and return everything
        (wbox, system, reference_force, force, integrator, ghmc, langevin) = self._make_ghmc_system(nprop, temperature)

        # Create the context
        platform = openmm.Platform.getPlatformByName('CUDA')
        properties = {'CUDAPrecision': 'mixed'}
        context = openmm.Context(system, integrator, platform, properties)
        context.setPositions(wbox.positions)

        # Take a few steps of langevin dynamics to test the compound integrator
        langevin.step(50)

        #### NCMC ####
        # Accumalating the work performed 3 ways.
        ext_work_integrator = 0.0    # Externally-accumulated unitless work calculated using integrator variables
        ext_work_getenergy = 0.0     # Externally-accumulated unitless work calculated using getEnergy()
        # The third work term is accumulated inside the integrator.

        # Steps are taken with GHMC, again making use of the compound integrator
        ghmc.step(1)
        for stage in range(npert):
            # Initial energy with getEnergy
            initial_energy_getenergy = context.getState(getEnergy=True).getPotentialEnergy() / kT
            # Initial energy from integrator
            initial_energy_integrator = ghmc.getGlobalVariableByName('potential_new') / kT_unitless
            # Perturbation
            self._update_forces(stage, npert, force, reference_force)
            force.updateParametersInContext(context)
            # Accumulate work (a)
            final_energy_getenergy = context.getState(getEnergy=True).getPotentialEnergy() / kT
            ext_work_getenergy += (final_energy_getenergy - initial_energy_getenergy)
            # Propagation
            ghmc.step(1)
            # Accumulate work (b)
            final_energy_integrator = ghmc.getGlobalVariableByName('potential_initial') / kT_unitless
            ext_work_integrator += (final_energy_integrator - initial_energy_integrator)

        # Both work estimates should agree.
        assert ( abs(ext_work_getenergy - ext_work_integrator) < 0.01 )

    @pytest.mark.skipif(detect_opencl(), reason="OpenCL not detected on platform")
    def test_ghmc_integrator_opencl(self):
        """
        Tests the GHMC integrator with an NCMC procedure on a box of water. The protocol work is calculated inside the
        integrator and externally with getState(getEnergy=True). The work calculated with both methods should agree.

        A compound integrator is used to make this test more like the simulation procedure with saltswap.

        This version runs the test on CUDA
        """
        temperature = 298.0 * unit.kelvin

        # NCMC parameters
        npert = 128          # Number of perturbation steps
        nprop = 2           # Number of propagation steps per perturbation

        # Get thermal energy
        (kT, kT_unitless) = self._get_constants(temperature)

        # Make the water box test system and return everything
        (wbox, system, reference_force, force, integrator, ghmc, langevin) = self._make_ghmc_system(nprop, temperature)

        # Create the context
        platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        context = openmm.Context(system, integrator, platform, properties)
        context.setPositions(wbox.positions)

        # Take a few steps of langevin dynamics to test the compound integrator
        langevin.step(50)

        #### NCMC ####
        # Accumalating the work performed 3 ways.
        ext_work_integrator = 0.0    # Externally-accumulated unitless work calculated using integrator variables
        ext_work_getenergy = 0.0     # Externally-accumulated unitless work calculated using getEnergy()
        # The third work term is accumulated inside the integrator.

        # Steps are taken with GHMC, again making use of the compound integrator
        ghmc.step(1)
        for stage in range(npert):
            # Initial energy with getEnergy
            initial_energy_getenergy = context.getState(getEnergy=True).getPotentialEnergy() / kT
            # Initial energy from integrator
            initial_energy_integrator = ghmc.getGlobalVariableByName('potential_new') / kT_unitless
            # Perturbation
            self._update_forces(stage, npert, force, reference_force)
            force.updateParametersInContext(context)
            # Accumulate work (a)
            final_energy_getenergy = context.getState(getEnergy=True).getPotentialEnergy() / kT
            ext_work_getenergy += (final_energy_getenergy - initial_energy_getenergy)
            # Propagation
            ghmc.step(1)
            # Accumulate work (b)
            final_energy_integrator = ghmc.getGlobalVariableByName('potential_initial') / kT_unitless
            ext_work_integrator += (final_energy_integrator - initial_energy_integrator)

        # Both work estimates should agree.
        assert ( abs(ext_work_getenergy - ext_work_integrator) < 0.01 )