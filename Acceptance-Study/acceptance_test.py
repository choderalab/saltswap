from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from openmmtools import integrators
from datetime import datetime
import sys
sys.path.append("../saltswap/")
import saltswap

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an openmm simulation with salt exchange moves.")
    parser.add_argument('-o','--out',type=str,help="the filename of the PDB where configurations will be deposited",default="output.pdb")
    parser.add_argument('-d','--data',type=str,help="the filename of the text file where the simulation data will be stored",default="data.txt")
    parser.add_argument('-u','--deltachem',type=float,help="the difference between the chemical potential in kJ/mol of water and salt, default=-700",default=-700.0)
    parser.add_argument('-c','--cycles',type=int,help="the number of cycles between MD and MCMC salt-water swaps, default=100",default=200)
    parser.add_argument('-s','--steps',type=int,help="the number of MD steps per cycle, default=25000",default=25000)
    parser.add_argument('-a','--attempts',type=int,help="the number of salt-water swap moves attempted, default=100",default=100)
    parser.add_argument('-e','--equilibration',type=int,help="the number of equilibration steps, default=1000",default=1000)
    parser.add_argument('--nkernals',type=int,help="the number of NCMC kernals, default=1000",default=1000)
    parser.add_argument('--nverlet',type=int,help="the number of Verlet steps used in each NCMC iteration, default=1",default=1)
    parser.add_argument("--gpu",action='store_true',help="whether the simulation will be run on a GPU, default=False",default=False)
    args = parser.parse_args()

# Setting the parameters of the simulation
size = 20.0*unit.angstrom     # The length of the edges of the water box.
temperature = 300*unit.kelvin
pressure = 1*unit.atmospheres
delta_chem = args.deltachem*unit.kilojoule_per_mole

# Creating the test system
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME)

# Creating an integrator for both MD and NCMC
compound_integrator = openmm.CompoundIntegrator()
#compound_integrator.addIntegrator(openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds))
compound_integrator.addIntegrator(integrators.GHMCIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds))
compound_integrator.addIntegrator(integrators.VelocityVerletIntegrator(1.0*unit.femtoseconds))
compound_integrator.setCurrentIntegrator(0)

# Add pressure.
wbox.system.addForce(openmm.MonteCarloBarostat(pressure, temperature, 25))

# Check if doing GPU or CPU and create the context
if args.gpu==False :
    platform = openmm.Platform.getPlatformByName('CPU')
    context = openmm.Context(wbox.system, compound_integrator,platform)
else :
    platform = openmm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}
    context = openmm.Context(wbox.system, compound_integrator, platform, properties)
context.setPositions(wbox.positions)
context.setVelocitiesToTemperature(temperature)

# Quick equilibration
compound_integrator.step(args.equilibration)

# Creating the saltswap object
mc_saltswap = saltswap.SaltSwap(system=wbox.system,topology=wbox.topology,temperature=temperature,delta_chem=delta_chem,integrator=compound_integrator,pressure=pressure,debug=False,nkernals=args.nkernals, nverlet_steps=args.nverlet)

# Opening file to store simulation data
f = open(args.data, 'w')
s = "Niterations = {:4}, Nsteps = {:7}, Nattemps = {:3}, Dchem = {:5}, Nkernals = {:5}, Nverlet = {:3}\n".format(args.cycles,args.steps,args.attempts,args.deltachem,args.nkernals,args.nverlet)
f.write(s)
s = "\n{:4} {:5} {:5} {:4} {:6}\n".format("Step","Nwats","Nsalt","AccProb","Time (s)")
f.write(s)
f.close()

# Opening a PDB file to store configurations
positions = context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
pdbfile = open(args.out, 'w')
app.PDBFile.writeHeader(wbox.topology, file=pdbfile)
app.PDBFile.writeModel(wbox.topology, positions, file=pdbfile, modelIndex=0)

iterations = args.cycles          # Number of rounds of MD and constant salt moves
nsteps = args.steps               # Amount of MD steps per iteration. 250000 steps = 500 picoseconds
nattempts = args.attempts         # Number of identity exchanges for water and ions.

# Running simulation
startTime = datetime.now()
for i in range(iterations):
    iter_start = datetime.now()
    compound_integrator.step(nsteps)
    mc_saltswap.update(context,nattempts=nattempts)
    iter_time = datetime.now() - iter_start
    # Custom reporters: (simulations.reporters severely slows the simulations down)
    cnts = mc_saltswap.getIdentityCounts()
    nrg = mc_saltswap.getPotEnergy(context)
    acc = mc_saltswap.getAcceptanceProbability()
    f = open(args.data, 'a')
    s = "{:4} {:5} {:5}   {:0.2f} {:4}\n".format(i,cnts[0],cnts[1],round(acc,2),iter_time.seconds)
    f.write(s)
    f.close()
    positions = context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    app.PDBFile.writeModel(wbox.topology, positions, file=pdbfile, modelIndex=i+1)
tm = datetime.now() - startTime

f = open(args.data, 'a')
s = "\nElapsed time in seconds = {:7}".format(tm.seconds)
f.write(s)
s = "\nNumber of NaNs = {:3}\n".format(mc_saltswap.nan)
f.write(s)


