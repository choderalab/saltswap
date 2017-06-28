from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app
import numpy as np
from saltswap import wrappers
from openmmtools import integrators
import saltswap.record as Record


def get_initial_bias(fn, max_salt):
    """
    Get decent estimates of the initial biases for SAMS by via least squares fitting of previously calculated relative free energies.
    Using the model
        y = c + m*log(x + 1)
    where y is the relative free energy to add salt, x is the number of salt present; c and m are to be determined.

    This allows us to extrapolate previously calculated free energies to salt numbers that have not been simulatied.

    Parameters
    ----------
    fn: numpy.ndarray
        the cumulative free energy to add salt from zero to some amount. The length of fn = max_salt + 1, as
        the first value is the free energy to add no salt.
    max_salt: int
        the maximum number of salt for which free energies will be produced.

    Returns
    -------
    cumulative_prediction: numpy.ndarray
        the predicted cumulative free energy to add salt
    """
    # Get the relative free energies
    y = np.diff(fn)
    x = np.log(np.arange(len(y)) + 1.0)

    # Perform least squares fitting
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]

    # Use the fitted parameters to predict the cumulative free energies to add salt for the supplied values.
    relative_prediction = c + m * np.log(np.arange(max_salt) + 1.0)
    cumulative_prediction = np.hstack((0.0, np.cumsum(relative_prediction)))

    return cumulative_prediction


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

    # SAMS parameters
    saltmin = 0
    nstates = args.saltmax - saltmin + 1
    target_weights = np.repeat(1./float(nstates), nstates)
    two_stage = True    # Whether to do burn-in stage
    beta = 0.7          # Exponent for burn-in stage
    precision = 0.2     # How close the sampling proportions must be to the target weights before the burn-in finishes.

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

    # Initialize SAMS adaptor with previous estimates
    if args.model == 'tip3p':
        fn = wrappers.default_tip3p_weights['fn']
        initial_bias = get_initial_bias(fn, args.saltmax)
    elif args.model == 'tip4pew':
        fn = wrappers.default_tip4pew_weights['fn']
        initial_bias = get_initial_bias(fn, args.saltmax)
    else:
        fn = (wrappers.default_tip4pew_weights['fn'] + wrappers.default_tip3p_weights['fn']) / 2.0
        initial_bias = get_initial_bias(fn, args.saltmax)

    # Create the sams salinator object for the calculating free energies for the insertion and deletion of salt
    sams_salinator = wrappers.SAMSSalinator(saltmin=0, saltmax=args.saltmax, initial_bias=initial_bias,
                                            two_stage=two_stage, beta=beta, target_weights=target_weights,
                                            precision=precision, context=context, system=wbox.system,
                                            topology=wbox.topology, ncmc_integrator=ncmc_langevin,
                                            salt_concentration=0.1 * unit.molar, pressure=pressure,
                                            temperature=temperature, npert=npert, water_name='HOH')
    # Thermalize the system
    langevin.step(args.equilibration)

    # Create the netcdf file for non-configuration simulation data
    filename = args.out + '.nc'
    creator = Record.CreateNetCDF(filename)
    simulation_control_parameters = {'timestep': timestep, 'splitting': splitting, 'box_edge': box_edge,
                                     'collision_rate': collision_rate}
    ncfile = creator.create_netcdf(sams_salinator.swapper, simulation_control_parameters, nstates=nstates)

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

    # The actual simulation
    k = 0
    for iteration in range(args.iterations):
        # Propagate configurations and salt concentrations, and update SAMS bias.
        langevin.step(args.steps)
        sams_salinator.update()

        # Save data
        if iteration % args.save_freq == 0:
            # Record the simulation data
            Record.record_netcdf(ncfile, context, sams_salinator.swapper, k, attempt=0, sams_bias=sams_salinator.bias, sync=True)

            if args.save_configs:
                # Record the simulation configurations
                positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
                dcd.writeModel(positions=positions)
            k += 1
