from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app

import numpy as np

from saltswap import wrappers,
from openmmtools import integrators
import saltswap.record as Record
from bams.sams_adapter import SAMSAdaptor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a saltswap simulation on a box of water.")
    parser.add_argument('-o','--out', type=str,
                        help="the naming scheme of the output results, default=out", default="out")
    parser.add_argument('-b','--box_edge', type=float,
                        help="length of the water box edge in Angstroms, default=30", default=30.0)
    parser.add_argument('-i','--iterations', type=int,
                        help="the number of iterations of MD and saltswap moves, default=10000", default=10000)
    parser.add_argument('-s','--steps', type=int,
                        help="the number of MD steps per iteration, default=2000", default=2000)
    parser.add_argument('--save_freq', type=int,
                        help="the frequency with which to save the data", default=4)
    parser.add_argument('--timestep', type=float,
                        help='the timestep of the integrators in femtoseconds, default=2.0', default=2.0)
    parser.add_argument('-e','--equilibration', type=int,
                        help="the number of equilibration steps, default=5000", default=5000)
    parser.add_argument('--model', type=str, choices=['tip3p','tip4pew'],
                        help="the water model, default=tip4ew", default='tip4pew')
    parser.add_argument('--npert', type=int,
                        help="the number of ncmc perturbation kernels, default=10000", default=10000)
    parser.add_argument('--saltmax', type=int,
                        help="the maximum number of salt pairs that will be accepted, default=20", default=20)
    parser.add_argument('--platform', type=str, choices=['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    parser.add_argument('--save_configs', action='store_true',
                        help="whether to save the configurations of the box of water, default=False", default=False)
    args = parser.parse_args()

    # Setting the parameters of the simulation
    timestep = args.timestep * unit.femtoseconds
    box_edge = args.box_edge * unit.angstrom
    npert = args.npert

    # Fixed simulation parameters
    splitting = 'V R O R V'
    temperature = 300.*unit.kelvin
    collision_rate = 1./unit.picoseconds
    pressure = 1.*unit.atmospheres

    # Make the water box test system with a fixed pressure
    wbox = WaterBox(model=args.model, box_edge=box_edge, nonbondedMethod=app.PME, cutoff=10*unit.angstrom, ewaldErrorTolerance=1E-4)
    wbox.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    # Create the compound integrator
    langevin = integrators.LangevinIntegrator(splitting=splitting, temperature=temperature, timestep=timestep,
                                              collision_rate=collision_rate, measure_shadow_work=False,
                                              measure_heat=False)
    ncmc_langevin = integrators.ExternalPerturbationLangevinIntegrator(splitting=splitting, temperature=temperature,
                                                                       timestep=timestep, collision_rate=collision_rate,
                                                                       measure_shadow_work=False, measure_heat=False)
    integrator = openmm.CompoundIntegrator()
    integrator.addIntegrator(langevin)
    integrator.addIntegrator(ncmc_langevin)

    # Create context
    if args.platform == 'CUDA':
        platform = openmm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        properties = {'CudaPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, platform, properties)
    elif args.platform == 'OpenCL':
        platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, platform, properties)
    elif args.platform == 'CPU':
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(wbox.system, integrator, platform)
    else:
        raise Exception('Platform name {0} not recognized.'.format(args.platform))

    context.setPositions(wbox.positions)
    context.setVelocitiesToTemperature(temperature)

    # Create the swapper object for the insertion and deletion of salt
    # Create the salinator object for the insertion and deletion of salt
    sams_salinator = wrappers.SAMSSalinator(context=context, system=wbox.system, topology=wbox.topology,
                                   ncmc_integrator=ncmc_langevin, salt_concentration=0.1*unit.molar,
                                   pressure=pressure, temperature=temperature, npert=npert, water_name='HOH', )

    mcdriver = Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=0.0,
                        ncmc_integrator=ncmc_langevin, pressure=pressure, npert=npert, nprop=1)

    # Thermalize the system
    langevin.step(args.equilibration)

    # Create the netcdf file for non-configuration simulation data
    filename = args.out + '.nc'
    creator = Record.CreateNetCDF(filename)
    simulation_control_parameters = {'timestep': timestep, 'splitting': splitting, 'box_edge': box_edge,
                                     'collision_rate': collision_rate}
    ncfile = creator.create_netcdf(mcdriver, simulation_control_parameters)
    # Create an additional dimension to store sams bias
    ncfile.createDimension('nsalt', args.saltmax + 1)
    ncfile.groups['Sample state data'].createVariable('sams bias', 'f8', ('iteration', 'nsalt'), zlib=True)

    if args.save_configs:
        # Create PDB file to view with the (binary) dcd file.
        positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
        pdbfile = open(args.out + '.pdb', 'w')
        app.PDBFile.writeHeader(wbox.topology, file=pdbfile)
        app.PDBFile.writeModel(wbox.topology, positions, file=pdbfile, modelIndex=0)
        pdbfile.close()

        # Create a DCD file system configurations
        dcdfile = open(args.out + '.dcd', 'wb')
        dcd = app.DCDFile(file=dcdfile, topology=wbox.topology, dt=timestep)

    # Initialize SAMS adaptor
    initial_guess = 317.0
    # Initializing the  bias using the free energy to insert a single salt molecule
    bias = np.arange(args.saltmax + 1) * initial_guess
    adaptor = SAMSAdaptor(nstates=args.saltmax + 1, zetas=bias, beta=0.6, flat_hist=0.1)
    state_counts = np.zeros(args.saltmax + 1)

    # Proposal mechanism for SAMS
    def gen_penalty(nsalt, bias, saltmax):
        if nsalt == saltmax:
            penalty = [0.0, bias[nsalt - 1] - bias[nsalt]]
        elif nsalt == 0:
            penalty = [bias[nsalt + 1] - bias[nsalt], 0.0]
        else:
            penalty = [bias[nsalt + 1] - bias[nsalt],
                       bias[nsalt - 1] - bias[nsalt]]
        return penalty

    # The actual simulation
    k = 0
    for iteration in range(args.iterations):
        # Establish the free energy penalty for adding or removing more salt
        nwater, ncation, nanion = mcdriver.get_identity_counts()
        penalty = gen_penalty(ncation, bias, args.saltmax)

        # Propagate configurations and salt concentrations
        langevin.step(args.steps)
        mcdriver.attempt_identity_swap(context, penalty=penalty, saltmax=args.saltmax)

        # Update SAMS estimator
        nwater, ncation, nanion = mcdriver.get_identity_counts()
        noisy = np.zeros(args.saltmax + 1)
        noisy[ncation] = 1
        state_counts[ncation] += 1
        bias = adaptor.update(state=ncation, noisy_observation=noisy, histogram=state_counts)

        # Save data
        if iteration % args.save_freq == 0:
            # Record the simulation data
            Record.record_netcdf(ncfile, context, mcdriver, k, attempt=0, sync=False)
            ncfile.groups['Sample state data']['sams bias'][k, :] = bias
            ncfile.sync()
            if args.save_configs:
                # Record the simulation configurations
                positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
                dcd.writeModel(positions=positions)
            k += 1
