import numpy as np
from saltswap.sams_adapter import SAMSAdaptor

class TestSAMSAdaptor(object):
    """
    Functions to test the SAMS adaptor class
    """

    def test_sams_clock(self):
        """
        Assess that the adaptor accurately logs the number of times it was called and its iteration stage. The SAMS
        internal clock is used to calculate the gain. Samples will be generated with numpy's multinomial sampler.
        """
        # Initialize the SAMS adaptor
        nstates = 10
        adaptor = SAMSAdaptor(nstates=nstates)

        # The probabilities for the discrete states
        probs = np.repeat(1.0 / nstates, nstates)

        # Iterating the sampler and adaptor for few steps
        niterations = 10
        for i in range(niterations):
            noisy = np.random.multinomial(1, probs)
            state = np.where(noisy != 0)[0][0]
            adaptor.update(state=state, noisy_observation=noisy)

        assert adaptor.time == niterations

    def test_slow_gain(self):
        """
        Ensure the correct gain factor is correctly calculated in the absence of two stage procedure.
        """
        # The system and SAMS parameters
        nstates = 10          # The number of discrete states in the system
        target_weights = np.repeat(1.0 / nstates, nstates)      # The target probabilities
        initial_zeta = np.repeat(1.0, nstates)                  # Initial estimate for the free energies
        initial_zeta -= initial_zeta[0]

        adaptor = SAMSAdaptor(nstates=nstates, two_stage=False, zetas=initial_zeta, target_weights=target_weights)

        # Generate a fake sample for the noisy variable:
        noisy = np.zeros(nstates)
        state = 3             # The imagined state of the sampler
        noisy[state] = 1.0

        new_zetas = adaptor.update(state=state, noisy_observation=noisy)
        true_new_zeta = initial_zeta[state] + (noisy[state] / target_weights[state]) / adaptor.time

        assert new_zetas[state] == true_new_zeta

    def test_fast_gain(self):
        """
        Ensures the gain during the burn-in stage is properly calculated.
        """
        # The system and SAMS parameters
        nstates = 10          # The number of discrete states in the system
        beta = 0.7            # Exponent of gain factor during burn-in phase.
        target_weights = np.repeat(1.0 / nstates, nstates)      # The target probabilities
        initial_zeta = np.repeat(1.0, nstates)                  # Initial estimate for the free energies
        initial_zeta -= initial_zeta[0]

        adaptor = SAMSAdaptor(nstates=nstates, two_stage=True, zetas=initial_zeta, target_weights=target_weights,
                              beta=beta)

        # Generate a fake sample for the noisy variable:
        noisy = np.zeros(nstates)
        state = 3  # The imagined state of the sampler
        noisy[state] = 1.0

        new_zetas = adaptor.update(state=state, noisy_observation=noisy, histogram=np.arange(nstates))

        if target_weights[state] < adaptor.time**(-beta):
            true_new_zeta = initial_zeta[state] + target_weights[state] * (noisy[state] / target_weights[state])
        else:
            true_new_zeta = initial_zeta[state] + adaptor.time**(-beta) * (noisy[state] / target_weights[state])

        assert new_zetas[state] == true_new_zeta

    def test_burnin_continuation(self):
        """
        Tests whether the burn-in does not stop prematurely
        """
        nstates = 10
        target_weights = np.repeat(1.0 / nstates, nstates)
        adaptor = SAMSAdaptor(nstates=nstates, two_stage=True, flat_hist=0.2, target_weights=target_weights)

        # Creating a histogram that is far from 'flat' and therefore should not stop the burn-in phase
        histogram = np.arange(nstates)

        # Creating a fake sample
        noisy = np.zeros(nstates)
        state = 3  # The imagined state of the sampler
        noisy[state] = 1.0

        adaptor.update(state=state, noisy_observation=noisy, histogram=histogram)

        assert adaptor.burnin == True

    def test_burnin_termination(self):
        """
        Tests whether the stopping criterion for the burn-in period is correctly implemented.
        """
        nstates = 10
        target_weights = np.repeat(1.0 / nstates, nstates)
        adaptor = SAMSAdaptor(nstates=nstates, two_stage=True, flat_hist=0.2, target_weights=target_weights)

        # Creating a histogram that matches the target weights. The burn-in should stop.
        histogram = 100 * target_weights

        noisy = np.zeros(nstates)
        state = 3  # The imagined state of the sampler
        noisy[state] = 1.0

        adaptor.update(state=state, noisy_observation=noisy, histogram=histogram)

        assert adaptor.burnin == False
