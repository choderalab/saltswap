import numpy as np
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from saltswap.perturbator import Perturbator


"""
Scrip to run the Perturbator class on a box of water
"""

def create_example(stage=0, nstages=20, mode='add salt', size=25.0, platform='CPU', temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres):
    """
    Create an alchemical perturbation object for a box of water. The outputs will allow the computation of free energies
    and gradients along an alchemical salt insertion/deletion path.

    Parameters
    ----------
    stage: int
        the alchemical stage that will be simulated
    nstages: int
        the total number of alchemical intermediates
    mode: str
        whether the relative free energies will be calculated in the direction of adding or removing salt.
        Either 'add salt' or 'remove salt'
    size: float
        the length of one side of the water box in Angstroms
    platform: str
        The openmm platform on which the simulation will be run. Either CPU, CUDA or OpenCL.
    temperature: simtk.unit
        the temperature of the simulation
    pressure: simtk.unit
        the pressure of the simulation

    Returns
    -------
    perturber: saltswap.Perturber
        the object that can alter the non-bonded parameters of water and salt and calculate gradients and energies
    integrator: simtk.openmm.integrator
        the openmm object that can propagate the dynamics
    """
    # Create a box of water
    wbox = WaterBox(box_edge=size * unit.angstrom, nonbondedMethod=app.PME, cutoff=10 * unit.angstrom, ewaldErrorTolerance=1E-4)
    integrator = openmm.LangevinIntegrator(temperature, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
    barostat = openmm.MonteCarloBarostat(pressure, temperature)
    wbox.system.addForce(barostat)

    # Creating the context:
    if platform == 'CUDA':
        platform = openmm.Platform.getPlatformByName(platform)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        properties = {'CudaPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, platform, properties)
    elif platform == 'OpenCL':
        platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, platform, properties)
    else:
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(wbox.system, integrator, platform)
    context.setPositions(wbox.positions)
    context.setVelocitiesToTemperature(temperature)

    perturber = Perturbator(topology=wbox.topology, system=wbox.system, integrator=integrator, context=context,
                            mode=mode, stage=stage, nstages=nstages, temperature=temperature, pressure=pressure)

    return perturber, integrator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an alchemical perturbations of water to salt.")
    parser.add_argument('-o','--out',type=str,
                        help="the naming scheme of the numpy file saves",default="alchemical")
    parser.add_argument('-i','--iterations', type=int,
                        help="the iterations of MD",default=200)
    parser.add_argument('-s','--steps', type=int,
                        help="the number of MD steps per iteration, default=2000", default=2000)
    parser.add_argument('-e','--equilibration', type=int,
                        help="the number of equilibration steps, default=500000", default=500000)
    parser.add_argument('--size', type=float,
                        help="the length of the water box, default=25.0", default=25.0)
    parser.add_argument('--state', type=int,
                        help="the index of the alchemical that will be simulated", default=0)
    parser.add_argument('--nstates', type=int,
                        help="the number of alchemical intermediates", default=20)
    parser.add_argument('--platform', type=str, choices = ['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()

    # Initialize
    perturber, integrator = create_example(stage=args.state, nstages=args.nstates, mode='add salt', size=args.size,
                                           platform=args.platform)

    # Equilibrate
    integrator.step(args.equilibration)

    # Pre-assignment
    alchemical_energies = np.zeros((args.iterations, args.nstates))
    alchemical_gradients = np.zeros((args.iterations, args.nstates))

    # Run simulation and record energies and gradients
    for i in range(args.iterations):
        integrator.step(args.steps)
        # Save the energies at each alchemical state that have been calculated with the configuration at the current state
        energies = perturber.perturb_all_states(in_thermal_units=True)
        alchemical_energies[i,:] = np.array(energies)
        # Save the gradient at the current state
        alchemical_gradients[i,:] = perturber.gradients_all_stages(in_thermal_units=True)
        # Save data every 10 iterations
        if i % 10 == 0:
            np.save(args.out + '_energies', alchemical_energies)
            np.save(args.out + '_grads', alchemical_gradients)

    # Save data for the last time
    np.save(args.out + '_energies', alchemical_energies)
    np.save(args.out + '_grads', alchemical_gradients)