from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from datetime import datetime
#import sys
#sys.path.append("/Users/rossg/Work/saltswap/SAMS")
from calibrate import MCMCSampler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an openmm simulation with salt exchange moves.")
    parser.add_argument('-o','--out',type=str,help="the filename of the PDB where configurations will be deposited",default="output.pdb")
    parser.add_argument('-d','--data',type=str,help="the filename of the text file where the simulation data will be stored",default="data.txt")
    parser.add_argument('-u','--deltachem',type=float,help="the difference between the chemical potential in kJ/mol of water and salt, default=700",default=700.0)
    parser.add_argument('-c','--cycles',type=int,help="the number of cycles between MD and MCMC salt-water swaps, default=100",default=200)
    parser.add_argument('-s','--steps',type=int,help="the number of MD steps per cycle, default=25000",default=1000)
    parser.add_argument('-a','--attempts',type=int,help="the number of salt-water swap moves attempted, default=100",default=10)
    parser.add_argument('-e','--equilibration',type=int,help="the number of equilibration steps, default=1000",default=1000)
    parser.add_argument('--npert',type=int,help="the number of NCMC perturbation kernels, default=1000",default=1000)
    parser.add_argument('--nprop',type=int,help="the number of propagation kernels per perturbation, default=1",default=1)
    parser.add_argument('--propagator',type=str,help="the type integrator used for propagation in NCMC, default=GHMC",default='GHMC')
    parser.add_argument("--gpu",action='store_true',help="whether the simulation will be run on a GPU, default=False",default=False)
    args = parser.parse_args()


    # Setting the parameters of the simulation
    size = 20.0*unit.angstrom     # The length of the edges of the water box.
    temperature = 300*unit.kelvin
    pressure = 1*unit.atmospheres
    delta_chem = args.deltachem*unit.kilojoule_per_mole

    if args.gpu == True:
        ctype = 'CUDA'
    else:
        ctype = 'CPU'

    # Creating the test system
    wbox = WaterBox(box_edge=size,cutoff=9*unit.angstrom,nonbondedMethod=app.PME)

    # Initialize the class that can sample over MD and salt-water exchanges.
    sampler = MCMCSampler(wbox.system, wbox.topology, wbox.positions, temperature=temperature, pressure=pressure, nkernels=args.npert,
                          nverlet=args.nprop, propagator=args.propagator, delta_chem=delta_chem, mdsteps=args.steps, saltsteps=args.attempts, ctype=ctype)

    # Thermalize
    sampler.gen_config(mdsteps=args.equilibration)

    # Opening file to store simulation data
    f = open(args.data, 'w')
    s = "Niterations = {:4}, Nsteps = {:7}, Nattemps = {:3}, Dchem = {:5}, Nperts = {:5}, Nprops = {:3}\n".format(args.cycles,args.steps,args.attempts,args.deltachem,args.npert,args.nprop)
    f.write(s)
    s = "\n{:4} {:5} {:5} {:4} {:6}\n".format("Step","Nwats","Nsalt","AccProb","Time (s)")
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

    iterations = args.cycles          # Number of rounds of MD and constant salt moves
    # Running simulation
    startTime = datetime.now()
    for i in range(iterations):
        iter_start = datetime.now()
        sampler.move()
        iter_time = datetime.now() - iter_start
        # Saving acceptance probability data:
        cnts = sampler.saltswap.getIdentityCounts()
        nrg = sampler.saltswap.getPotEnergy(sampler.context)
        acc = sampler.saltswap.getAcceptanceProbability()
        f = open(args.data, 'a')
        s = "{:4} {:5} {:5}   {:0.2f} {:4}\n".format(i,cnts[0],cnts[1],round(acc,2),iter_time.seconds)
        f.write(s)
        f.close()
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
    tm = datetime.now() - startTime

    f = open(args.data, 'a')
    s = "\nElapsed time in seconds = {:7}".format(tm.seconds)
    f.write(s)
    s = "\nNumber of NaNs = {:3}\n".format(sampler.saltswap.nan)
    f.write(s)
