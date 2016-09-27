# Loading modules and pretty colours
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
import sys
sys.path.append("../saltswap/")
from mcmc_samplers import MCMCSampler
import numpy as np
import matplotlib.pyplot as plt
from pymbar import timeseries as ts
from time import time

############################################
############################################
# Nice colours, taken on 26th Nov 2015 from:
#http://tableaufriction.blogspot.co.uk/2012/11/finally-you-can-use-tableau-data-colors.html

# These are the "Tableau" colors as RGB. I've chosen my faves.
# In order: blue, green, purple, orange. Hopefully a good compromise for colour-blind people.
tableau4 = [(31, 119, 180),(44, 160, 44),(148,103,189),(255, 127, 14)]
tableau4_light = [(174,199,232),(152,223,138),(197,176,213),(255,187,120)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau4)):
    r, g, b = tableau4[i]
    tableau4[i] = (r / 255., g / 255., b / 255.)
    r, g, b = tableau4_light[i]
    tableau4_light[i] = (r / 255., g / 255., b / 255.)
############################################
############################################

# Creating the test system
size = 15.0*unit.angstrom
temperature = 300*unit.kelvin
pressure = 1*unit.atmospheres
cutoff = size/2 - 0.5*unit.angstrom   # The electrostatic cutoff. Must be at least half the system size

# Simulation parameters
mus = [-1.0, 0.0, 1.0]      # The applied (delta) chemical potentials
Nsamps = 200                # The number of sampling blocks
nattempts = 20              # The number of insertion/deletion attempts in a sampling block
skip = 50
# NCMC specific parameters
nprop = 1
npert = 5

# Pre-assigment
r_inst = np.zeros(len(mus))      # mean ratio of water to salt
r_ghmc = np.zeros(len(mus))
r_vv = np.zeros(len(mus))

rstd_inst = np.zeros(len(mus))   # standard deviation of water to salt
rstd_ghmc = np.zeros(len(mus))
rstd_vv = np.zeros(len(mus))


#########################
# Instanteneous switching
#########################
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=cutoff)
t = time()
for i in range(len(mus)):
    dummystate = MCMCSampler(wbox.system,wbox.topology,wbox.positions,delta_chem=mus[i])
    dummystate.saltswap.cation_parameters = dummystate.saltswap.water_parameters
    dummystate.saltswap.anion_parameters = dummystate.saltswap.water_parameters
    Nwats = []
    Nsalts = []
    ratio = []
    for block in range(Nsamps):
        dummystate.gen_label(saltsteps=nattempts)
        (nwats,nsalt,junk) = dummystate.saltswap.getIdentityCounts()
        ratio.append(1.0*nwats/nsalt)
        Nwats.append(nwats)
        Nsalts.append(nsalt)
    ratio = np.array(ratio)
    r_inst[i] = ratio[skip:].mean()
    rstd_inst[i] = ratio[skip:].std()*np.sqrt(ts.statisticalInefficiency(ratio[skip:]))
print 'Instanstaneous switching test took {0:f} seconds'.format(time() - t)

#########################
# NCMC with GHMC
#########################
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=cutoff)
t = time()
for i in range(len(mus)):
    dummystate = MCMCSampler(wbox.system,wbox.topology,wbox.positions,delta_chem=mus[i],npert=npert,nprop=nprop,propagator='GHMC')
    dummystate.saltswap.cation_parameters = dummystate.saltswap.water_parameters
    dummystate.saltswap.anion_parameters = dummystate.saltswap.water_parameters
    Nwats = []
    Nsalts = []
    ratio = []
    for block in range(Nsamps):
        dummystate.gen_label(saltsteps=nattempts)
        (nwats,nsalt,junk) = dummystate.saltswap.getIdentityCounts()
        ratio.append(1.0*nwats/nsalt)
        Nwats.append(nwats)
        Nsalts.append(nsalt)
    ratio = np.array(ratio)
    r_ghmc[i] = ratio[skip:].mean()
    rstd_ghmc[i] = ratio[skip:].std()*np.sqrt(ts.statisticalInefficiency(ratio[skip:]))
print 'NCMC with GHMC test took {0:f} seconds'.format(time() - t)

#########################
# NCMC with GHMC
#########################
wbox = WaterBox(box_edge=size,nonbondedMethod=app.PME,cutoff=cutoff)
t = time()
for i in range(len(mus)):
    dummystate = MCMCSampler(wbox.system,wbox.topology,wbox.positions,delta_chem=mus[i],npert=npert,nprop=nprop,propagator='velocityVerlet')
    dummystate.saltswap.cation_parameters = dummystate.saltswap.water_parameters
    dummystate.saltswap.anion_parameters = dummystate.saltswap.water_parameters
    Nwats = []
    Nsalts = []
    ratio = []
    for block in range(Nsamps):
        dummystate.gen_label(saltsteps=nattempts)
        (nwats,nsalt,junk) = dummystate.saltswap.getIdentityCounts()
        ratio.append(1.0*nwats/nsalt)
        Nwats.append(nwats)
        Nsalts.append(nsalt)
    ratio = np.array(ratio)
    r_vv[i] = ratio[skip:].mean()
    rstd_vv[i] = ratio[skip:].std()*np.sqrt(ts.statisticalInefficiency(ratio[skip:]))
print 'NCMC with velocity Verlet test took {0:f} seconds'.format(time() - t)
