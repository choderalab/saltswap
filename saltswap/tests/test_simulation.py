from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app
import numpy as np
from saltswap import swapper, mcmc_samplers
from openmmtools import integrators as open_integrators
from saltswap.integrators import GHMCIntegrator

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

        salinator = swapper.Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=0.0,
                integrator=integrator, pressure=pressure, npert=npert, nprop=1)

        return integrator, context, salinator

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
        salinator = swapper.Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=0.0,
                integrator=integrator, pressure=None, npert=npert, nprop=1)

        return integrator, context, salinator

    def test_initialize(self):
        """
        Tests the basic initialization of a Swapper object with a saltswap GHMC integrator and Langevin integrator.
        """
        integrator, context, salinator =  self._create_langevin_system()
        integrator, context, salinator = self._create_ghmc_system()

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

        integrator, context, salinator = self._create_langevin_system()

        nsteps = 10         # Number of steps of equilibrium MD
        niterations = 5     # Number of MD and salt concentration updates
        nattempts = 1       # Number insertion/deletion attempts per iteration

        external_work = np.zeros((niterations, salinator.npert + 1))
        internal_work = np.zeros((niterations, salinator.npert + 1))
        for i in range(niterations):
            integrator.setCurrentIntegrator(0)
            integrator.step(nsteps)
            integrator.setCurrentIntegrator(1)
            salinator.update(context, nattempts=nattempts)
            internal_work[i,:], external_work[i,:] = salinator.compare_protocol_work(context)

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

        integrator, context, salinator = self._create_ghmc_system()

        nsteps = 10         # Number of steps of equilibrium MD
        niterations = 5     # Number of MD and salt concentration updates
        nattempts = 1       # Number insertion/deletion attempts per iteration

        external_work = np.zeros((niterations, salinator.npert + 1))
        internal_work = np.zeros((niterations, salinator.npert + 1))
        for i in range(niterations):
            integrator.setCurrentIntegrator(0)
            integrator.step(nsteps)
            integrator.setCurrentIntegrator(1)
            salinator.update(context, nattempts=nattempts)
            internal_work[i,:], external_work[i,:] = salinator.compare_protocol_work(context)

        print(internal_work - external_work)
        assert np.all(np.absolute(internal_work - external_work) < 1E-4)

    def test_langevin_mcmc_sampler(self):
        """
        Testing the basic functionality of the MCMC wrapper 'MCMCSampler' with the openmmtools Langevin integrator.
        """
        wbox = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic)
        sampler = mcmc_samplers.MCMCSampler(wbox.system, wbox.topology, wbox.positions, propagator='Langevin',
                                            delta_chem=0, nprop=1, npert=5)
        sampler.multimove(nmoves=2, mdsteps=1, saltsteps=1)


    def test_ghmc_mcmc_sampler(self):
        """
        Testing the basic functionality of the MCMC wrapper 'MCMCSampler' with the saltswap GHMC integrator.
        """
        wbox = WaterBox(nonbondedMethod=openmm.app.CutoffPeriodic)
        sampler = mcmc_samplers.MCMCSampler(wbox.system, wbox.topology, wbox.positions, propagator='GHMC',
                                            delta_chem=0, nprop=1, npert=5)
        sampler.multimove(nmoves=2, mdsteps=1, saltsteps=1)

