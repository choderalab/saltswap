from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from time import time
from saltswap.mcmc_samplers import MCMCSampler
import numpy as np

"""
Command line tool for testing the application of a chemical potential to a box of water.
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an openmm simulation with salt exchange moves on a box of water.")
    parser.add_argument('-o','--out', type=str,
                        help="the naming scheme for the output structure files" ,default="output")
    parser.add_argument('-d','--data', type=str,
                        help="the filename of the text file where the simulation data will be stored", default="data.txt")
    parser.add_argument('-u','--deltachem', type=float,
                        help="the applied chemical potential in thermal units, default=335", default=335.0)
    parser.add_argument('-c','--cycles', type=int,
                        help="the number of cycles between MD and MCMC salt-water swaps, default=100", default=200)
    parser.add_argument('-s','--steps', type=int,
                        help="the number of MD steps per cycle, default=25000", default=2000)
    parser.add_argument('-a','--attempts', type=int,
                        help="the number of salt-water swap moves attempted, default=100", default=10)
    parser.add_argument('-e','--equilibration', type=int,
                        help="the number of equilibration steps, default=1000", default=1000)
    parser.add_argument('--saltmax', type=int,
                        help="the maximum number of salt pairs that will be inserted, default=None",default=None)
    parser.add_argument('--npert', type=int,
                        help="the number of _ncmc perturbation kernels, default=1000", default=1000)
    parser.add_argument('--nprop', type=int,
                        help="the number of propagation kernels per perturbation, default=1", default=1)
    parser.add_argument('--timestep', type=float,
                        help='the _ncmc propagator timstep in femtoseconds, default=1.0', default=1.0)
    parser.add_argument('--propagator', type=str,
                        help="the type integrator used for propagation in _ncmc, default=GHMC",default='GHMC')
    parser.add_argument('--platform', type=str, choices=['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()


    # Setting the parameters of the simulation
    size = 30.0  *unit.angstrom     # The length of the edges of the water box.
    temperature = 300 * unit.kelvin
    pressure = 1 * unit.atmospheres
    timestep = args.timestep * unit.femtoseconds

    # Creating the test system, with non-bonded switching function and lower than standard PME error tolerance
    wbox = WaterBox(box_edge=size, nonbondedMethod=app.PME, cutoff=10*unit.angstrom, ewaldErrorTolerance=1E-4)

    # Initialize the class that can sample over MD and salt-water exchanges.
    sampler = MCMCSampler(wbox.system, wbox.topology, wbox.positions, temperature=temperature, pressure=pressure,
                          saltmax=args.saltmax, npert=args.npert, nprop=args.nprop, propagator=args.propagator,
                          timestep=timestep, delta_chem=args.deltachem, mdsteps=args.steps,
                          saltsteps=args.attempts, platform=args.platform)

    ghmc_equilibrium = sampler.integrator.getIntegrator(0)
    ghmc_ncmc= sampler.integrator.getIntegrator(1)

    # Opening file to store simulation data
    f = open(args.data, 'w')
    s = "Niterations = {:4}, Nsteps = {:7}, Nattemps = {:3}, Dchem = {:5}, Nperts = {:5}, Nprops = {:3}\n".format(args.cycles, args.steps, args.attempts, args.deltachem, args.npert, args.nprop)
    f.write(s)
    s = "\n{:4} {:5} {:5} {:4} {:9} {:9} {:6}\n".format("Step","Nwats","Nsalt","AccProb", "GHMC_NCMC_acc", "GHMC_Equil_Acc", "Time (s)")
    f.write(s)
    f.close()

    # Thermalize
    sampler.gen_config(mdsteps=args.equilibration)
    ghmc_equilibrium.setGlobalVariableByName('naccept', 0)
    ghmc_equilibrium.setGlobalVariableByName('ntrials', 0)

    # Open PDB file for writing.
    pdbfile = open(args.out + '.pdb', 'w')
    app.PDBFile.writeHeader(wbox.topology, file=pdbfile)
    app.PDBFile.writeModel(wbox.topology, wbox.positions, file=pdbfile, modelIndex=0)
    pdbfile.close()

    # Create a DCD file
    positions = sampler.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
    dcdfile = open(args.out + '.dcd', 'wb')
    dcd = app.DCDFile(file=dcdfile, topology=wbox.topology, dt=args.timestep)
    dcd.writeModel(positions=positions)

    iterations = args.cycles          # Number of rounds of MD and constant salt moves
    # Running simulation
    startTime = time()
    for i in range(iterations):
        iter_start = time()
        sampler.move()
        iter_time = time() - iter_start
        # Saving acceptance probability data:
        cnts = sampler.saltswap.get_identity_counts()
        acc = sampler.saltswap.get_acceptance_probability()
        if args.nprop != 0 and args.propagator == 'GHMC':
            ghmc_acc = np.mean(np.array(sampler.saltswap.naccepted_ncmc_integrator))
            equil_acc = ghmc_equilibrium.getGlobalVariableByName('naccept') / ghmc_equilibrium.getGlobalVariableByName('ntrials')
        else:
            ghmc_acc = 0.0
            equil_acc = 0.0
        sampler.saltswap.naccepted_ncmc_integrator = []
        f = open(args.data, 'a')
        s = "{:4} {:5} {:5}   {:0.2f}      {:0.2f}         {:0.2f}         {:0.1f}\n".format(i, cnts[0], cnts[1], round(acc,2), round(ghmc_acc,2), round(equil_acc,2), iter_time)
        f.write(s)
        f.close()
        # Save trajectory
        positions = sampler.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
        dcd.writeModel(positions=positions)
    dcdfile.close()

    tm = time() - startTime

    f = open(args.data, 'a')
    s = "\nElapsed time in seconds = {:0.1f}".format(tm)
    f.write(s)
    s = "\nNumber of NaNs = {:3}\n".format(sampler.saltswap.nan)
    f.write(s)
