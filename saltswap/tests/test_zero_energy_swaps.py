import numpy as np
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.testsystems import WaterBox
from mcmc_samplers import MCMCSampler


class TestZeroEnergySwaps(object):
    """
    Test for swapping non-bonded parameters of the tip3p water model into the same tip3p model. As the energy difference
    between these swaps is zero, the proportion of labelled molecules is known exactly. A box of pure water is used for
    the tests.

    Swaps are attempted between 2 water molecules and a cation and anion (salt) pair. They are labelled as such, despite having
    the same parameters, as the proportion of water molecules, N_w, to salt molecules, N_s is given by

    N_w / N_s = exp( -( \beta \Delta \mu / 2 ),

    where  \beta \Delta \mu is the difference in the chemical potential of salt and water multiplied by the inverse
    temperature. (The factor of 1/2 in the exponent comes from the fact that two water molecules are exchanged for
    two ions.)
    """
    def test_ideal_swaps_instant(self, Dmu = 0, size = 15.0 * unit.angstrom, Nsamps = 50):
        """
        Box of water test for zero energy swap for instantaneous insertions and deletions.

        Parameter
        ---------
        Dmu : float
          The difference in chemical potential between 2 water molecules and 1 salt molecule, in units of kT.
        size : Quantity length
          The length of the sides of the water box
        Nsamps : int
          The number repeats for batches of insertion and deletion attempts
        """
        # Calculating the target ratio
        target = np.exp(- Dmu / 2)

        # Create the box of water to simulate
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        dummystate = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = Dmu, nprop = 0, npert = 1)
        dummystate.saltswap.cation_parameters = dummystate.saltswap.water_parameters    # Setting the cation parameters to water's
        dummystate.saltswap.anion_parameters = dummystate.saltswap.water_parameters     # Setting the anion parameters to water's
        dummystate.saltswap._set_parampath()     # Recalculating the peturbation path for new paramters

        # Sampling. Recording the ratio of water to salt every 20 attempts and repeated Nsamps times.
        ratio = []
        for batch in range(Nsamps):
            dummystate.gen_label(saltsteps=20)
            (nwats,nsalt,junk) = dummystate.saltswap.get_identity_counts()
            ratio.append(1.0*nwats/nsalt)
        ratio = np.array(ratio)
        ratio_mean = np.mean(ratio)
        ratio_std_error = np.std(ratio)/np.sqrt(Nsamps)

        # Seeing if the calculated ratio is close to the target ratio.
        # Increasing standard error by a factor 2.5, as correlations between samples underestimate the error
        assert (ratio_mean < target + 2.5 * ratio_std_error) and (ratio_mean > target - 2.5 * ratio_std_error)

    def test_ideal_swaps_GHMC(self, Dmu = 0, size = 15.0 * unit.angstrom, Nsamps = 50):
        """
        Box of water test for zero energy swap for _ncmc insertions and deletions using a GHMC integrator.

        Parameter
        ---------
        Dmu : float
          The difference in chemical potential between 2 water molecules and 1 salt molecule, in units of kT.
        size : Quantity length
          The length of the sides of the water box
        Nsamps : int
          The number repeats for batches of insertion and deletion attempts
        """
        # Calculating the target ratio
        target = np.exp(- Dmu / 2)

        # Create the box of water to simulate
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        dummystate = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = Dmu, nprop = 1, npert = 5, propagator='GHMC')
        dummystate.saltswap.cation_parameters = dummystate.saltswap.water_parameters    # Setting the cation parameters to water's
        dummystate.saltswap.anion_parameters = dummystate.saltswap.water_parameters     # Setting the anion parameters to water's
        dummystate.saltswap._set_parampath()     # Recalculating the peturbation path for new paramters

        # Sampling. Recording the ratio of water to salt every 20 attempts and repeated Nsamps times.
        ratio = []
        for batch in range(Nsamps):
            dummystate.gen_label(saltsteps=20)
            (nwats,nsalt,junk) = dummystate.saltswap.get_identity_counts()
            ratio.append(1.0*nwats/nsalt)
        ratio = np.array(ratio)
        ratio_mean = np.mean(ratio)
        ratio_std_error = np.std(ratio)/np.sqrt(Nsamps)

        # Seeing if the calculated ratio is close to the target ratio.
        # Increasing standard error by a factor 2.5, as correlations between samples underestimate the error
        assert (ratio_mean < target + 2.5 * ratio_std_error) and (ratio_mean > target - 2.5 * ratio_std_error)

    def test_ideal_swaps_velocity_verlet(self, Dmu = 0, size = 15.0 * unit.angstrom, Nsamps = 50):
        """
        Box of water test for zero energy swap for _ncmc insertions and deletions using a velocity Verlet integrator.

        Parameter
        ---------
        Dmu : float
          The difference in chemical potential between 2 water molecules and 1 salt molecule, in units of kT.
        size : Quantity length
          The length of the sides of the water box
        Nsamps : int
          The number repeats for batches of insertion and deletion attempts
        """
        # Calculating the target ratio
        target = np.exp(- Dmu / 2)

        # Create the box of water to simulate
        wbox = WaterBox(box_edge = size, nonbondedMethod = app.PME, cutoff = size/2 - 0.5*unit.angstrom)
        dummystate = MCMCSampler(wbox.system, wbox.topology, wbox.positions, delta_chem = Dmu, nprop = 1, npert = 5, propagator='velocityVerlet')
        dummystate.saltswap.cation_parameters = dummystate.saltswap.water_parameters    # Setting the cation parameters to water's
        dummystate.saltswap.anion_parameters = dummystate.saltswap.water_parameters     # Setting the anion parameters to water's
        dummystate.saltswap._set_parampath()     # Recalculating the peturbation path for new paramters

        # Sampling. Recording the ratio of water to salt every 20 attempts and repeated Nsamps times.
        ratio = []
        for batch in range(Nsamps):
            dummystate.gen_label(saltsteps=20)
            (nwats,nsalt,junk) = dummystate.saltswap.get_identity_counts()
            ratio.append(1.0*nwats/nsalt)
        ratio = np.array(ratio)
        ratio_mean = np.mean(ratio)
        ratio_std_error = np.std(ratio)/np.sqrt(Nsamps)

        # Seeing if the calculated ratio is close to the target ratio.
        # Increasing standard error by a factor 2.5, as correlations between samples underestimate the error
        assert (ratio_mean < target + 2.5 * ratio_std_error) and (ratio_mean > target - 2.5 * ratio_std_error)