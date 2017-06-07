from simtk import openmm, unit
from openmmtools.testsystems import WaterBox, DHFRExplicit
from simtk.openmm import app
import numpy as np
from saltswap import swapper, wrappers
from openmmtools import integrators as open_integrators
from saltswap.integrators import GHMCIntegrator



class TestSalinator(object):
    """
    Tests the functionality of wrapper for the MCMC addition and deletion of salt.
    """

    def _create_protein_system(self, npert=10):
        """
        Creates a DHFR system to re-use in tests.
        """
        # System parameters
        temperature = 300. * unit.kelvin
        timestep = 2. * unit.femtoseconds
        collision_rate = 90. / unit.picoseconds
        pressure = 1. * unit.atmospheres
        salt_concentration = 0.2 * unit.molar

        # Make the water box test system with a fixed pressure
        box = DHFRExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)
        box.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

        # Create the compound integrator with SaltSwap's custom GHMC integrator
        langevin = open_integrators.LangevinIntegrator(splitting="V R O R V", temperature=temperature,
                                                       timestep=timestep, collision_rate=collision_rate)
        ncmc_langevin = open_integrators.ExternalPerturbationLangevinIntegrator(splitting="V R O R V",
                                                                                temperature=temperature,
                                                                                timestep=timestep,
                                                                                collision_rate=collision_rate)
        integrator = openmm.CompoundIntegrator()
        integrator.addIntegrator(langevin)
        integrator.addIntegrator(ncmc_langevin)

        # Create context
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(box.system, integrator, platform)
        context.setPositions(box.positions)

        salinator = wrappers.Salinator(context=context, system=box.system, topology=box.topology,
                                       ncmc_integrator=ncmc_langevin, salt_concentration=salt_concentration,
                                       pressure=pressure, temperature=temperature, npert=npert, water_name='WAT')

        return integrator, context, salinator

    def _create_waterbox(self, model='tip4pew', npert=10):
        """
        Creates a DHFR system to re-use in tests.
        """
        # System parameters
        temperature = 300. * unit.kelvin
        timestep = 2. * unit.femtoseconds
        collision_rate = 90. / unit.picoseconds
        pressure = 1. * unit.atmospheres
        salt_concentration = 0.2 * unit.molar

        # Make the water box test system with a fixed pressure
        # Make the water box test system with a fixed pressure
        box = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic, model=model)
        box.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))


        # Create the compound integrator with SaltSwap's custom GHMC integrator
        langevin = open_integrators.LangevinIntegrator(splitting="V R O R V", temperature=temperature,
                                                       timestep=timestep, collision_rate=collision_rate)
        ncmc_langevin = open_integrators.ExternalPerturbationLangevinIntegrator(splitting="V R O R V",
                                                                                temperature=temperature,
                                                                                timestep=timestep,
                                                                                collision_rate=collision_rate)
        integrator = openmm.CompoundIntegrator()
        integrator.addIntegrator(langevin)
        integrator.addIntegrator(ncmc_langevin)

        # Create context
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(box.system, integrator, platform)
        context.setPositions(box.positions)

        salinator = wrappers.Salinator(context=context, system=box.system, topology=box.topology,
                                       ncmc_integrator=ncmc_langevin, salt_concentration=salt_concentration,
                                       pressure=pressure, temperature=temperature, npert=npert, water_name='HOH')

        return integrator, context, salinator

    def test_protein_initialization(self):
        """
        Test the initialization of the salinator
        """
        integrator, context, salinator = self._create_protein_system()

    def test_waterbox_initialization(self):
        """
        Implicitly tests the identification of tip3p and tip4p water models, as exceptions are thrown if the water
        model is not recognized during the class initialization.
        """
        integrator, context, salinator = self._create_waterbox(model='tip4pew')
        integrator, context, salinator = self._create_waterbox(model='tip3p')

    def test_invert_concentration(self):
        """
        Make sure that when calculating the chemical potential from the concentration, the step that involves inverting
        the chemical potential-->concentration is correctly performed.
        """
        integrator, context, salinator = self._create_waterbox(model='tip4pew')

        # Get the default weights for tip4pew water
        weights = wrappers.default_tip4pew_weights
        fn = weights['fn']
        volume = weights['volume']

        # Get the concentration for a specified chemical potential
        delta_chem = 315.0
        predicted_concentration = salinator.predict_concentration(delta_chem, fn, volume)

        # Get the chemical potential from the concentration
        predicted_delta_chem = salinator.invert_concentration(predicted_concentration, fn, volume)

        assert abs(predicted_delta_chem - delta_chem) < 1E-6

    def test_charge_calculation(self):
        """
        Verify type of total charge calculation.
        """
        integrator, context, salinator = self._create_protein_system()

        nonbonded_force = salinator._get_nonbonded_force()
        initial_charge = salinator._get_system_charge(nonbonded_force)

        # There should be a non-zero total charge for the DFHR test system
        assert type(initial_charge) == int and initial_charge != 0

    def test_neutralizer(self):
        """
        Make sure the system can be correctly neutralized.
        """
        integrator, context, salinator = self._create_protein_system()

        # Neutralize
        salinator.neutralize()

        # Get the new charge
        nonbonded_force = salinator._get_nonbonded_force()
        final_charge = salinator._get_system_charge(nonbonded_force)

        assert final_charge == 0

    def test_initial_insertion(self):
        """
        Test to make sure salt gets added when an initial guess of the number is made.
        """
        integrator, context, salinator = self._create_waterbox()

        # Neutralize
        salinator.neutralize()

        # Get the initial number of salt
        initial_nsalt = np.sum(salinator.swapper.stateVector == 1)

        # Add salt to the specified concentration
        salinator.initialize_concentration()

        # Get the final number of salt
        final_nsalt = np.sum(salinator.swapper.stateVector == 1)

        assert final_nsalt > initial_nsalt

        # Ensure that the system remains neutral.
        nonbonded_force = salinator._get_nonbonded_force()
        final_charge = salinator._get_system_charge(nonbonded_force)
        assert final_charge == 0

    def test_update(self):
        """
        Test the ability for the wrapper to perform an instantaneous insertion/deletion move.
        """
        # Set number of perturbations to zero for instantaneous insertions and deletions.
        integrator, context, salinator = self._create_waterbox(npert=1)

        salinator.update(nattempts=1)


class TestWaterBoxSimulation(object):
    """
    Tests the basic running of SaltSwap on a box of water.
    """
    def _create_langevin_system(self):
        """
        Creates the box of water as a test system with a compound Langevin integrator.

        Returns
        -------
        integrator: openmm.CompoundIntegrator
            the compound integrator to perform equilibrium and nonequilibrium sampling
        context: openmm.Context
            the openmm context
        salinator: saltswap.swapper
            the driver that can perform salt insertion and deletion moves
        """
        # System parameters
        temperature = 300. * unit.kelvin
        timestep = 2. * unit.femtoseconds
        collision_rate = 90. / unit.picoseconds
        pressure = 1. * unit.atmospheres
        npert = 10

        # Make the water box test system with a fixed pressure
        wbox = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic)
        wbox.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

        # Create the compound integrator with SaltSwap's custom GHMC integrator
        langevin = open_integrators.LangevinIntegrator(splitting="V R O R V", temperature=temperature, timestep=timestep, collision_rate=collision_rate)
        ncmc_langevin = open_integrators.ExternalPerturbationLangevinIntegrator(splitting="V R O R V", temperature=temperature, timestep=timestep, collision_rate=collision_rate)
        integrator = openmm.CompoundIntegrator()
        integrator.addIntegrator(langevin)
        integrator.addIntegrator(ncmc_langevin)

        # Create context
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(wbox.system, integrator, platform)
        context.setPositions(wbox.positions)

        driver = swapper.Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=0.0,
                                    ncmc_integrator=ncmc_langevin, pressure=pressure, npert=npert, nprop=1)

        return integrator, context, driver

    def _create_ghmc_system(self):
        """
        Creates the box of water as a test system with a compound Langevin integrator.

        Returns
        -------
        integrator: openmm.CompoundIntegrator
            the compound integrator to perform equilibrium and nonequilibrium sampling
        context: openmm.Context
            the openmm context
        salinator: saltswap.swapper
            the driver that can perform salt insertion and deletion moves
        """
        # System parameters
        temperature = 300. * unit.kelvin
        timestep = 2. * unit.femtoseconds
        collision_rate = 90. / unit.picoseconds
        pressure = 1. * unit.atmospheres
        npert = 10

        # Make the water box test system with a fixed pressure
        wbox = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic)
        # TODO: this test fails when a barostat is added to the system. The integrator requires a fix.
        #wbox.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

        # Create the compound integrator with SaltSwap's custom GHMC integrator
        ghmc = GHMCIntegrator(temperature=temperature, timestep=timestep, collision_rate=collision_rate, nsteps=1)
        ncmc_ghmc = GHMCIntegrator(temperature=temperature, timestep=timestep, collision_rate=collision_rate, nsteps=1)
        integrator = openmm.CompoundIntegrator()
        integrator.addIntegrator(ghmc)
        integrator.addIntegrator(ncmc_ghmc)

        # Create context
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(wbox.system, integrator, platform)
        context.setPositions(wbox.positions)

        # TODO: set pressure in Swapper when issue with GHMC barostat is fixed.
        driver = swapper.Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=0.0,
                                    ncmc_integrator=ncmc_ghmc, pressure=None, npert=npert, nprop=1)

        return integrator, context, driver

    def test_initialize(self):
        """
        Tests the basic initialization of a Swapper object with a saltswap GHMC integrator and Langevin integrator.
        """
        integrator, context, driver =  self._create_langevin_system()
        integrator, context, driver = self._create_ghmc_system()

    def test_combine_mc_md(self):
        """
        Tests the combination equilibrium configuration sampling and counterion concentration sampling with saltswap.
        """
        integrator, context, salinator = self._create_ghmc_system()

        nsteps = 10         # Number of steps of equilibrium MD
        niterations = 5     # Number of MD and salt concentration updates
        nattempts = 1       # Number insertion/deletion attempts per iteration

        for i in range(niterations):
            integrator.setCurrentIntegrator(0)
            integrator.step(nsteps)
            integrator.setCurrentIntegrator(1)
            salinator.update(context, nattempts=nattempts)

    def test_ncmc_protocol_work_langevin(self):
        """
        Make sure the protocol work that is accumulated by the custom integrator agrees with the work calculated with
        getState().
        """
        # Use only one CPU thread to ensure determinism of energy evaluations
        import os
        os.environ['OPENMM_CPU_THREADS'] = '1'

        integrator, context, driver = self._create_langevin_system()

        nsteps = 10         # Number of steps of equilibrium MD
        niterations = 5     # Number of MD and salt concentration updates
        nattempts = 1       # Number insertion/deletion attempts per iteration

        external_work = np.zeros((niterations, driver.npert + 1))
        internal_work = np.zeros((niterations, driver.npert + 1))
        for i in range(niterations):
            integrator.setCurrentIntegrator(0)
            integrator.step(nsteps)
            integrator.setCurrentIntegrator(1)
            driver.update(context, nattempts=nattempts)
            internal_work[i,:], external_work[i,:] = driver.compare_protocol_work(context)

        print(internal_work[-1,:])
        print(external_work[-1, :])
        assert np.all(np.absolute(internal_work - external_work) < 1E-4)

    def test_ncmc_protocol_work_ghmc(self):
        """
        Make sure the protocol work that is accumulated by the custom integrator agrees with the work calculated with
        getState().
        """
        # Use only one CPU thread to ensure determinism of energy evaluations
        import os
        os.environ['OPENMM_CPU_THREADS'] = '1'

        integrator, context, driver = self._create_ghmc_system()

        nsteps = 10         # Number of steps of equilibrium MD
        niterations = 5     # Number of MD and salt concentration updates
        nattempts = 1       # Number insertion/deletion attempts per iteration

        external_work = np.zeros((niterations, driver.npert + 1))
        internal_work = np.zeros((niterations, driver.npert + 1))
        for i in range(niterations):
            integrator.setCurrentIntegrator(0)
            integrator.step(nsteps)
            integrator.setCurrentIntegrator(1)
            driver.update(context, nattempts=nattempts)
            internal_work[i,:], external_work[i,:] = driver.compare_protocol_work(context)

        print(internal_work - external_work)
        assert np.all(np.absolute(internal_work - external_work) < 1E-4)

    def test_langevin_mcmc_sampler(self):
        """
        Testing the basic functionality of the MCMC wrapper 'MCMCSampler' with the openmmtools Langevin integrator.
        """
        wbox = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic)
        sampler = wrappers.MCMCSampler(wbox.system, wbox.topology, wbox.positions, propagator='Langevin',
                                            delta_chem=0, nprop=1, npert=5)
        sampler.multimove(nmoves=2, mdsteps=1, saltsteps=1)


    def test_ghmc_mcmc_sampler(self):
        """
        Testing the basic functionality of the MCMC wrapper 'MCMCSampler' with the saltswap GHMC integrator.
        """
        wbox = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic)
        sampler = wrappers.MCMCSampler(wbox.system, wbox.topology, wbox.positions, propagator='GHMC',
                                            delta_chem=0, nprop=1, npert=5)
        sampler.multimove(nmoves=2, mdsteps=1, saltsteps=1)

