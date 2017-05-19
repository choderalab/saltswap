from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app

from saltswap.swapper import Swapper
from openmmtools import integrators
from saltswap.record import CreateNetCDF, record_netcdf, sample_variables

from netCDF4 import Dataset
import numpy as np

class TestRecorder(object):
    def create_waterbox_example(self):
        """
        Create a example system to simulate and record.
        """
        temperature = 300. * unit.kelvin
        timestep = 2. * unit.femtoseconds
        collision_rate = 90. / unit.picoseconds
        pressure = 1. * unit.atmospheres
        box_edge = 20.0 * unit.angstrom
        npert = 10
        splitting = 'V R R R O R R R V'

        # Make the water box test system with a fixed pressure
        wbox = WaterBox(box_edge=box_edge, nonbondedMethod=app.PME, cutoff=9 * unit.angstrom, ewaldErrorTolerance=1E-4)
        wbox.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

        # Create the compound integrator
        langevin = integrators.LangevinIntegrator(splitting=splitting, temperature=temperature, timestep=timestep,
                                                  collision_rate=collision_rate)
        ncmc_langevin = integrators.ExternalPerturbationLangevinIntegrator(splitting=splitting, temperature=temperature,
                                                                           timestep=timestep,
                                                                           collision_rate=collision_rate)
        integrator = openmm.CompoundIntegrator()
        integrator.addIntegrator(langevin)
        integrator.addIntegrator(ncmc_langevin)

        # Create context
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(wbox.system, integrator, platform)
        context.setPositions(wbox.positions)

        salinator = Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=0.0,
                                    ncmc_integrator=ncmc_langevin, pressure=pressure, npert=npert, nprop=1)

        return langevin, context, salinator


    def test_write_read(self):
        """
        Test that saving data occurs without error and can be loaded without error.
        """
        langevin, context, salinator = self.create_waterbox_example()

        # Control parameters for the simulation
        simulation_control_parameters = {'timestep': 2*unit.femtoseconds, 'splitting':'V R R R O R R R V',
                                         'box_edge': 20.0*unit.angstrom, 'collision_rate': 90./unit.picoseconds}

        # Create the netcdf file
        filename = 'test_recorder.nc'
        creator = CreateNetCDF(filename)
        ncfile = creator.create_netcdf(salinator, simulation_control_parameters)

        niterations = 3
        nattempts = 2
        for iteration in range(niterations):
            langevin.step(10)
            for attempt in range(nattempts):
                salinator.update(context, nattempts=1)
                record_netcdf(ncfile, context, salinator, iteration, attempt=attempt, sync=True)

        # Extract and store the results for easy testing
        saved_results = []
        for var in sample_variables:
            saved_results.append(ncfile.groups['Sample state data'][var][:])

        # Close netcdf file
        ncfile.close()

        # Re-open netcdf file and make sure it's the same as before
        opened_ncfile = Dataset(filename, 'r')

        # Make sure the uploaded numerical values are the same as before
        for var, res in zip(sample_variables, saved_results):
            assert np.all(opened_ncfile.groups['Sample state data'][var][:] == res)

