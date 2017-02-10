"""
    Example implementation of a simulation of a box of water where the number of counterions can fluctuate in a box of water using the semi-grand canonical ensemble.
"""
from datetime import datetime
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.integrators import VelocityVerletIntegrator
import cProfile
import sys
sys.path.append("../saltswap/")
import swapper
import gc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an openmm simulation with salt exchange moves.")
    parser.add_argument('-i','--input',type=str,help="the filename of the PDB structure of the starting configuration",default="waterbox.pdb")
    parser.add_argument('-o','--out',type=str,help="the filename of the PDB structure of the starting configuration",default="output.pdb")
    parser.add_argument('-d','--data',type=str,help="the filename of the text file where the simulation data will be stored",default="data.txt")
    parser.add_argument('-u','--deltachem',type=float,help="the difference between the chemical potential in kJ/mol of water and salt, default=-650",default=-650.0)
    parser.add_argument('-c','--cycles',type=int,help="the number of cycles between MD and MCMC salt-water swaps, default=200",default=200)
    parser.add_argument('-s','--steps',type=int,help="the number of MD steps per cycle, default=250000",default=250000)
    parser.add_argument('-a','--attempts',type=int,help="the number of salt-water swap moves attempted, default=100",default=100)
    parser.add_argument('-e','--equilibration',type=int,help="the number of equilibration steps, default=1000",default=1000)
    parser.add_argument('--nkernals',type=int,help="the number of _ncmc kernals, default=1000",default=1000)
    parser.add_argument('--nverlet',type=int,help="the number of Verlet steps used in each _ncmc iteration, default=1",default=1)
    parser.add_argument("--gpu",action='store_true',help="whether the simulation will be run on a GPU, default=False",default=False)
    parser.add_argument("--profile",action='store_true',help="whether each MD-MC iteration will be profiled, default=False",default=False)
    args = parser.parse_args()

#err = open("error.txt",'w')

# CONSTANTS
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
pressure = 1*unit.atmospheres
temperature = 300*unit.kelvin
delta_chem = args.deltachem*unit.kilojoule_per_mole

#Loading a premade water box:
pdb = app.PDBFile(args.input)
forcefield = app.ForceField('tip3p.xml')
system = forcefield.createSystem(pdb.topology,nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)

s = "Minimizing energy..."
print s
integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
context = openmm.Context(system, integrator)
context.setPositions(pdb.positions)
openmm.LocalEnergyMinimizer.minimize(context, 1.0, 25)
positions = context.getState(getPositions=True).getPositions(asNumpy=True)
del context, integrator

print "Creating compound integrator"
#lange_inte = openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
#vv_inte = VelocityVerletIntegrator(1.0*units.femtoseconds)
compound_integrator = openmm.CompoundIntegrator()
compound_integrator.addIntegrator(openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds))
compound_integrator.addIntegrator(VelocityVerletIntegrator(1.0*unit.femtoseconds))
compound_integrator.setCurrentIntegrator(0)

system.addForce(openmm.MonteCarloBarostat(pressure, temperature, 25))

s = "Initializing constant salt class"
print s
mc_saltswap = swapper.Swapper(system=system, topology=pdb.topology, temperature=temperature, delta_chem=delta_chem, integrator=compound_integrator, pressure=pressure, debug=False, nkernals=args.nkernals, nverlet_steps=args.nverlet)

s = "Initializing context"
print s
if args.gpu==False :
    platform = openmm.Platform.getPlatformByName('CPU')
    context = openmm.Context(system, compound_integrator,platform)
else :
    platform = openmm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed'}
    context = openmm.Context(system, compound_integrator, platform, properties)
context.setPositions(positions)

iterations = args.cycles          # Number of rounds of MD and constant salt moves
nsteps = args.steps               # Amount of MD steps per iteration. 250000 steps = 500 picoseconds
nattempts = args.attempts         # Number of identity exchanges for water and ions.

s = "Quick equilibration..."
print s
context.setVelocitiesToTemperature(temperature)
compound_integrator.step(args.equilibration)

positions = context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)

# Opening file to store simulation data
f = open(args.data, 'w')
s = "Niterations = {:4}, Nsteps = {:7}, Nattemps = {:3}, Dchem = {:5}, Nkernals = {:5}, Nverlet = {:3}\n".format(args.cycles,args.steps,args.attempts,args.deltachem,args.nkernals,args.nverlet)
f.write(s)
s = "\n{:4} {:5} {:5} {:4} {:6}\n".format("Step","Nwats","Nsalt","AccProb","Time (s)")
f.write(s)
f.close()
# Open PDB file for writing.
pdbfile = open(args.out, 'w')
app.PDBFile.writeHeader(pdb.topology, file=pdbfile)
app.PDBFile.writeModel(pdb.topology, positions, file=pdbfile, modelIndex=0)

def profile_bundle(nsteps,nattempts):
    compound_integrator.step(nsteps)
    mc_saltswap.update(context,nattempts=nattempts)

print "Running simulation..."
startTime = datetime.now()
for i in range(iterations):
    iter_start = datetime.now()
    if args.profile == False:
        compound_integrator.step(nsteps)
        mc_saltswap.update(context,nattempts=nattempts)
    else:
        filename = "profile{0}".format(i)
        cProfile.run("profile_bundle(nsteps,nattempts)",filename)
    iter_time = datetime.now() - iter_start
    # Custom reporters: (simulations.reporters severely slows the simulations down)
    cnts = mc_saltswap.get_identity_counts()
    nrg = mc_saltswap._get_potential_energy(context)
    dims = pdb.topology.getUnitCellDimensions()
    acc = mc_saltswap.get_acceptance_probability()
    f = open(args.data, 'a')
    s = "{:4} {:5} {:5}   {:0.2f} {:4}\n".format(i,cnts[0],cnts[1],round(acc,2),iter_time.seconds)
    f.write(s)
    f.close()
    mc_saltswap.reset_statistics()
    positions = context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    app.PDBFile.writeModel(pdb.topology, positions, file=pdbfile, modelIndex=i+1)
    gc.collect()
tm = datetime.now() - startTime

s = "\nElapsed time in seconds = {:7}".format(tm.seconds)
f.write(s)
s = "\nNumber of NaNs = {:3}\n".format(mc_saltswap.nan)
f.write(s)
