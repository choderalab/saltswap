#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""
Self adjusted mixture sampling for the SaltSwap class.

Description
-----------

Sampling the (semi) grand canonical distribution over the numbers of salt pairs.


Notes
-----

The is development.

References
----------

[1] Tan Z., Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics (17 November 2015)

Examples
--------

Coming soon to an interpreter near you!

TODO
----
    * Write the code

Copyright and license
---------------------

@author Gregory A. Ross <gregoryross.uk@gmail.com>

"""

import sys
import math
import random
import numpy as np
import simtk.openmm as openmm
import simtk.unit as units
import sys
sys.path.append("../saltswap/")
from saltswap import SaltSwap


class SAMS(SaltSwap):
    """
    Implementation of self-adjusted mixture sampling for exchanging water and salts in a grand canonical methodology

    Attributes
    ----------
    n_adaptations : int
        Number of times the relative free energies have been adapted.
    state_counts : np.array
        Histogram of the expected weights of current states.
    References
    ----------
    .. [1] Z. Tan, Optimally adjusted mixture sampling and locally weighted histogram analysis
        DOI: 10.1080/10618600.2015.1113975
    """

    def gen_config(self):
        """
        Generate a configuration via one of OpenMM's integrators
        :return:
        """
        pass

    def gen_label(self):
        """
        Generate a new number of salt molecules via SaltSwap's code
        :return:
        """

    def adapt_zeta(self):
        """
        Update the free energy estimate for the current state based SAMS binary procedure (equation 9)

        :return:
        """

    def move(self):
        self.gen_config()
        self.gen_label()
        self.adapt_zeta()

    def update(self,nmoves=None):

        for i in range(nmoves):
            self.move()
