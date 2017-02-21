from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox

"""
Script to time the functions 'setParticleParameters' and 'updateParametersInContext' in OpenMM by partially deleting
and re-instating a water molecule in a box of water molecules.
"""

def make_water_system(length=20.0, cutoff=9.0, platform='CUDA', equil_steps=10):
    """
    Create a box of TIP3P water and return the compenents of the OpenMM system.
    """
    wbox = WaterBox(box_edge=length * unit.angstrom, cutoff=cutoff * unit.angstrom, nonbondedMethod=app.PME)
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    if platform == 'CUDA':
        openmm_platform = openmm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, openmm_platform, properties)
    elif platform == 'OpenCL':
        openmm_platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, openmm_platform, properties)
    else:
        context = openmm.Context(wbox.system, integrator)
    context.setPositions(wbox.positions)
    context.setVelocitiesToTemperature(300*unit.kelvin)
    integrator.step(equil_steps)
    return context, integrator, wbox

def switch_water_force(force, context,frac=0.9):
    """
    Switch the non-bonded parameters of the first TIP3P water molecule to some specified fraction of the default values.
    """
    force.setParticleParameters(0, charge=-0.834*frac, sigma=0.3150752406575124*frac, epsilon=0.635968*frac)
    force.setParticleParameters(1, charge=0.417*frac, sigma=0, epsilon=1*frac)
    force.setParticleParameters(2, charge=0.417*frac, sigma=0, epsilon=1*frac)
    force.updateParametersInContext(context)

def run_timing(length=20.0, niterations=20, platform='CPU'):
    """
    Perform repeated switches of the non-bonded parameters of the first water molecule in the system
    """
    # Create the test system:
    context, integrator, wbox = make_water_system(length=length, platform=platform)
    # Get the non-bonded force:
    force = wbox.system.getForce(2)       # Non-bonded force.
    # Turn the first water molecule in the systen on and off for many iterations:
    for i in range(niterations):
        switch_water_force(force, context, frac=0.5)
        switch_water_force(force, context, frac=1.0)

if __name__ == "__main__":
    import argparse
    from cProfile import run
    parser = argparse.ArgumentParser(description="Run an openmm simulation with salt exchange moves.")
    parser.add_argument('-l','--length',type=float,
                        help="the length of one side of the water box in Angstroms, default=20.0", default=20.0)
    parser.add_argument('-n','--niterations',type=int,
                        help="the number of iterations to turn a water molecule on and off, default=20", default=10000)
    parser.add_argument('-o','--out',type=str, help="the filename of the text cProfile file", default="profile.pstat")
    parser.add_argument('--platform', type=str, choices = ['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()

    # Run cProfile on the non-bonded parameter switches on the water box
    run('run_timing(length=args.length, niterations=args.niterations, platform=args.platform)', filename=args.out)