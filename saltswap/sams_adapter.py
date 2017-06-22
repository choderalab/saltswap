import numpy as np

class SAMSAdaptor(object):
    """
    Implements the update scheme for self adjusted mixture sampling as described by Z. Tan in [1]. Can use either the
    Rao-Blackwellized or binary update schemes. To function, this class must be paired with a method to perform mixture
    sampling over states and configurations.

    [1] Journal of Computational and Graphical Statistics Vol. 26 , Iss. 1, 2017

    Example
    -------
    Calculating relative free energies of different states using the binary update scheme for samples drawn from a
    multinomial distribution. For this example, the IndependentMultinomialSampler, defined below, will be used. We know
    the true free energies in this case, but it serves to demonstrate the basic functionality this class. For this
    sampler class, we'll use the following as the true free energies:

    >>>  true_free_energies = np.array((0.0, -5.0, -10.0, -15.0))

    The free energies are relative to the first state. SAMS works by applying biases to each state, and updating these
    biases to achieve user specified sampling frequencies for each state. SAMS achieves by gradually improving estimates
    of the relative free energies of each state. These estimates are the biases used to achieve the specified sampling
    frequencies. We'll start off without any biases:

    >>> biases = np.zeros(len(true_free_energies))

    Now, we'll initialize the sampler:

    >>> sampler = IndependentMultinomialSampler(free_energies=true_free_energies, zetas=biases)

    Next, we'll initialize this SAMS update class, which will estimates given samples from states. We'll be using the
    two-stage scheme as described in equation 15 of [1], which will perform a burn-in for the free energies:

    >>> adaptor = SAMSAdaptor(nstates=len(true_free_energies), two_stage=True, flat_hist=0.2)

    The burn-in stage will finish when the state count histogram is within 20% (flat_hist=0.2) of the target weights,
    which specify the sampling frequencies we want. By default, the target_weights are uniform over the states.

    The SAMSAdaptor works by tracking the state of the sampler and updating the bias accordingly. The bias is used to
    sample from the different states at the target probability. There are 4 main stages to calculating free energies
    with this class. The example below performs 500 iterations of multinomial sampling and SAMS updates and each stage
    is labeled and described below.

    >>> for i in range(500):
    >>>     noisy = sampler.step()                 # 1.
    >>>     z = -adaptor.update(state=sampler.state , noisy_observation=noisy, histogram=sampler.state_counts)     # 3.
    >>>     sampler.zetas = z              # 4.

    In stage 1., states  sampled over with IndependentMultinomialSampler. The noisy variable is a binary vector with the
    only non-zero element at the index of the current state. In stage 3., the current state of the sampler, the noisy
    observation, and the observed state counts are supplied to the SAMS object. The state counts are not essential here,
    but are necessary if you want to use the two-stage scheme. The new bias required to achieve the target weights for
    that iteration is returned. In stage 4., the new bias is supplied to the sampler. Thus, the next iteration samples from
    the mixture with an updated state bias.

    As the number of iterations tends to infinity, the bias will converge to a value that can be used as an unbiased
    estimate of the free energies. When the target weights are equal, as they are above, the bias tends to
    true_free_energies.

    """
    def __init__(self, nstates, zetas=None, target_weights=None, two_stage=True, beta=0.6, flat_hist=0.2):
        """
        Parameters
        ----------
        nstates: int
            The number of free energies to infer
        zeta: numpy array
            The estimate of the free energy and the current state biasing potential
        target_weights: numpy array
            vector of the state probabilities that the sampler should converge to.
        two_stage: bool
            whether to perform the two-stage update procedure as outline by Z. Tan in Journal of Computational and
            Graphical Statistics Vol. 26 , Iss. 1, 2017. If true, the zeta parameters are adapted faster
        beta: float
            exponent of the gain during the bunr-in phase of the two-stage procedure. Should be between 0.5 and 1.0
        flat_hist: float
            degree of deviation that the state histogram can be from the target weights before the burn-in period
            in the two stage procedure ends. It is the maximum relative difference a histogram element can be
            from the respective target weight.
        """

        self.nstates = nstates
        self.beta = beta
        self.flat_hist = flat_hist
        self.two_stage = two_stage
        self.burnin = True
        self.time = 0
        self.burnin_length = None

        if zetas is None:
            self.zetas = np.zeros(self.nstates)
        elif len(zetas) != self.nstates:
            raise Exception('The length of the  bias/estimate (zetas) array is not equal to nstates')
        else:
            self.zetas = zetas

        if target_weights is None:
            self.target_weights = np.repeat(1.0 / nstates, nstates)
        else:
            if len(target_weights) != self.nstates:
                raise Exception('The length of the target weights array is not equal to nstates')
            elif np.abs(np.sum(target_weights) - 1.0) > 0.000001:
                raise Exception('The target weights do not sum to 1.')
            else:
                self.target_weights = target_weights

    def _calc_gain(self, state):
        """
        Calculates the gain factor for update.

        Parameter
        ---------
        state: int
            the index corresponding to the current state of the sampler

        Returns
        -------
        gain:
            the factor applied to the SAMS derived noisy variable
        """

        if self.two_stage:
            if self.burnin:
                gain = np.min((self.target_weights[state], self.time ** (-self.beta)))
            else:
                factor = (self.time - self.burnin_length + self.burnin_length ** (-self.beta)) ** (-1)
                gain = np.min((self.target_weights[state], factor))
        else:
            gain = 1.0 / self.time

        return gain

    def update(self, state, noisy_observation, histogram=None):
        """
        Update the estimate of the free energy based on the current state of the sample using either the binary or
        Rao-Blackwellized schemes, both of these schemes differ only by their noisy observable.

        Parameters
        ----------
        state: int
            the index corresponding to the current state of the sampler
        noisy_observation: numpy array
            the vector that will be multiplied by the gain factor when updating zeta
        histogram:
            the counts in each state collected over the simulation. Used to decide when to switch to the slow-growth
            stage if self.two_stage=True. If None, then slow-growth is automatically assumed.

        Returns
        -------
        zetas: numpy array
            the updated estimates for the free energies

        """
        # Ensure the internal clock is updated. Used for calculating the gain factor.
        self.time += 1

        if self.two_stage:
            if self.burnin and histogram is not None:
                # Calculate how far the histogram is from the target weights
                fraction = 1.0 * histogram / np.sum(histogram)
                dist = np.max(np.absolute(fraction - self.target_weights) / self.target_weights)
                if dist <= self.flat_hist:
                    # If histogram appears suitably flat then switch to slow growth
                    self.burnin = False
                    self.burnin_length = self.time
            elif self.burnin and histogram is None:
                # If no histogram is supplied the update scheme switches to slow growth
                self.burnin = False
                self.burnin_length = self.time

        gain = self._calc_gain(state)
        zetas_half = self.zetas + gain * (noisy_observation / self.target_weights)
        self.zetas = zetas_half - zetas_half[0]

        return self.zetas

class IndependentMultinomialSampler(object):
    """
    Class to draw independent samples from a biased multinomial distribution with the following distribution:

        p_i = exp(zeta_i - f_i) / sum_i[exp(zeta_i - f_i)]

    where f_i and zeta_i are the free energy and applied bias the free energy of ith state, respectively. The unbiased
    distribution has propabilities proportional to exp(-f_i).

    Example
    -------
    Specify the free energy difference between each state
    >>> free_energies = np.array((0.0, -10.0))

    Specify the biases to be applied to each state
    >>> biases = np.array((0.0, 0.0))

    Initialize the sampler
    >>> sampler = IndependentMultinomialSampler(free_energies=free_energies, zetas=biases)

    Take 200 uncorrelated global jumps over the states and record the current location in state_vector
    >>> state_vector = sampler.step(nsteps=200)

    The array state_vector is a binary vector with the only non-zero element at the current state of the system. The
    index of the non-zero element is given by
    >>> print(sampler.state)

    View how many times each state was visited over all the steps:
    >>> print(sampler.state_counts)
    """
    def __init__(self, free_energies=None, zetas=None):
        """
        Initialize the biased multinomial sampler.

        Parameters
        ----------
        free_energies: numpy array
            the free energy of the unbiased probabilities
        zetas: numpy array
            the exponent of the bias applied to the probabilities
        """
        if free_energies is None:
            self.free_energies = np.random.uniform(-50, 50, size=5)
            self.free_energies -= self.free_energies[0]
        else:
            self.free_energies = free_energies
            self.free_energies -= self.free_energies[0]

        if zetas is not None:
            self.zetas = zetas
            self.zetas -= self.zetas[0]
        else:
            self.zetas = np.repeat(0.0, len(self.free_energies))

        self.state_counts = np.repeat(0.0, len(self.free_energies))

        self.state = None

    def step(self, nsteps=1):
        """
        Sample from multinomial distribution and update the state histogram

        Parameters
        ----------
        nsteps: int
            the number of samples to draw

        Returns
        -------
        current_state: numpy array
            binary array where the only non-zero element indicates the current state of the system
        """
        p = np.exp(self.zetas - self.free_energies)
        p = p / np.sum(p)

        self.state_counts += np.random.multinomial(nsteps - 1, p)
        current_state = np.random.multinomial(1, p)
        self.state_counts += current_state
        self.state = int(np.where(current_state != 0)[0][0])

        return current_state

    def reset_statistics(self):
        """
        Reset the histogram state counter to zero
        """
        self.state_counts -= self.state_counts