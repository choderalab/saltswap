from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from time import time
from saltswap.mcmc_samplers import MCMCSampler
import numpy as np

"""
Command line tool for testing the combination of OpenMM molecule dynamics and Swapper water-salt exchanges. This script
has been designed to ease the testing of Swapper. It runs Swapper on a small box of water, and reports the statistics
for insertions and deletions, the work required to insert and delete salt, and snapshots of the structures.
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an openmm simulation with salt exchange moves on a box of water.")
    parser.add_argument('-o','--out',type=str,help="the filename of the PDB where configurations will be deposited",default="output.pdb")
    parser.add_argument('-d','--data',type=str,help="the filename of the text file where the simulation data will be stored",default="data.txt")
    parser.add_argument('-u','--deltachem',type=float,help="the difference between the chemical potential in kJ/mol of water and salt, default=700",default=700.0)
    parser.add_argument('-c','--cycles',type=int,help="the number of cycles between MD and MCMC salt-water swaps, default=100",default=200)
    parser.add_argument('-s','--steps',type=int,help="the number of MD steps per cycle, default=25000",default=1000)
    parser.add_argument('-a','--attempts',type=int,help="the number of salt-water swap moves attempted, default=100",default=10)
    parser.add_argument('-e','--equilibration',type=int,help="the number of equilibration steps, default=1000",default=1000)
    parser.add_argument('--npert',type=int,help="the number of _ncmc perturbation kernels, default=1000",default=1000)
    parser.add_argument('--nprop',type=int,help="the number of propagation kernels per perturbation, default=1",default=1)
    parser.add_argument('--timestep',type=float,help='the _ncmc propagator timstep in femtoseconds, default=1.0',default=1.0)
    parser.add_argument('--propagator',type=str,help="the type integrator used for propagation in _ncmc, default=GHMC",default='GHMC')
    parser.add_argument('--platform', type=str, choices = ['CPU','CUDA','OpenCL'],help="the platform where the simulation will be run, default=CPU",default='CPU')
    args = parser.parse_args()


    # Setting the parameters of the simulation
    size = 25.0*unit.angstrom     # The length of the edges of the water box.
    temperature = 300*unit.kelvin
    pressure = 1*unit.atmospheres
    delta_chem = args.deltachem*unit.kilojoule_per_mole

    # Creating the test system, with non-bonded switching function and lower than standard PME error tolerance
    wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=9*unit.angstrom,ewaldErrorTolerance=1E-5)

    # Initialize the class that can sample over MD and salt-water exchanges.
    timestep = args.timestep*unit.femtoseconds
    sampler = MCMCSampler(wbox.system, wbox.topology, wbox.positions, temperature=temperature, pressure=pressure, npert=args.npert,
                          nprop=args.nprop, propagator=args.propagator, ncmc_timestep = timestep, delta_chem=delta_chem, mdsteps=args.steps, saltsteps=args.attempts, platform=args.platform)

    # Thermalize
    sampler.gen_config(mdsteps=args.equilibration)

    # Opening file to store simulation data
    f = open(args.data, 'w')
    s = "Niterations = {:4}, Nsteps = {:7}, Nattemps = {:3}, Dchem = {:5}, Nperts = {:5}, Nprops = {:3}\n".format(args.cycles,args.steps,args.attempts,args.deltachem,args.npert,args.nprop)
    f.write(s)
    s = "\n{:4} {:5} {:5} {:4} {:9} {:6}\n".format("Step","Nwats","Nsalt","AccProb", "GHMC AccProb","Time (s)")
    f.write(s)
    f.close()

    # Opening a file to store work data for removing salt
    f = open("work_rm_"+args.data,"w")
    f.write("Total work required to remove salt per attempt\n")
    s = "Niterations = {:4}, Nsteps = {:7}, Nattemps = {:3}, Dchem = {:5}, Nperts = {:5}, Nprops= {:3}\n".format(args.cycles,args.steps,args.attempts,args.deltachem,args.npert,args.nprop)
    f.write(s)
    f.close()

    # Opening a file to store work data for adding salt
    f = open("work_add_"+args.data,"w")
    f.write("Total work required to add salt per attempt\n")
    s = "Niterations = {:4}, Nsteps = {:7}, Nattemps = {:3}, Dchem = {:5}, Nperts = {:5}, Nprops= {:3}\n".format(args.cycles,args.steps,args.attempts,args.deltachem,args.npert,args.nprop)
    f.write(s)
    f.close()

    # Open PDB file for writing.
    pdbfile = open(args.out, 'w')
    app.PDBFile.writeHeader(wbox.topology, file=pdbfile)
    app.PDBFile.writeModel(wbox.topology, wbox.positions, file=pdbfile, modelIndex=0)
    pdbfile.close()
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
            ghmc_acc = np.mean(np.array(sampler.saltswap.naccepted_ghmc))
        else:
            ghmc_acc = 0
        sampler.saltswap.naccepted_ghmc = []
        f = open(args.data, 'a')
        s = "{:4} {:5} {:5}   {:0.2f}      {:0.2f}       {:0.1f}\n".format(i, cnts[0], cnts[1], round(acc,2), round(ghmc_acc,2), iter_time)
        f.write(s)
        f.close()
        # Reset acceotance saltswap acceptance rate
        #sampler.saltswap.reset_statistics()
        # Saving work data for each of the nattempts and reseting:
        if len(sampler.saltswap.work_add) >= 0:
            f = open("work_add_"+args.data,"a")
            f.writelines("%s " % item  for item in sampler.saltswap.work_add)
            f.write("\n")
            f.close()
            sampler.saltswap.work_add=[]
        if len(sampler.saltswap.work_rm) >= 0:
            f = open("work_rm_"+args.data,"a")
            f.writelines("%s " % item  for item in sampler.saltswap.work_rm)
            f.write("\n")
            f.close()
            sampler.saltswap.work_rm=[]
        pdbfile = open(args.out, 'a')
        positions = sampler.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
        app.PDBFile.writeModel(wbox.topology, positions, file=pdbfile, modelIndex=i+1)
        pdbfile.close()

    tm = time() - startTime

    f = open(args.data, 'a')
    s = "\nElapsed time in seconds = {:0.1f}".format(tm)
    f.write(s)
    s = "\nNumber of NaNs = {:3}\n".format(sampler.saltswap.nan)
    f.write(s)
