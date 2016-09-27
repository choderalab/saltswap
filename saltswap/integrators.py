import numpy
import simtk.unit
import simtk.unit as units
import simtk.openmm as mm
kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

class GHMCIntegrator(mm.CustomIntegrator):

    """
    Generalized hybrid Monte Carlo (GHMC) integrator.

    """

    def __init__(self, temperature=298.0 * simtk.unit.kelvin, collision_rate=1.0 / simtk.unit.picoseconds, timestep=1.0 * simtk.unit.femtoseconds, nsteps=5):
        """
        Create a generalized hybrid Monte Carlo (GHMC) integrator.

        Parameters
        ----------
        temperature : simtk.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           The collision rate.
        timestep : simtk.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           The integration timestep.

        Notes
        -----
        This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
        Metrpolization step to ensure sampling from the appropriate distribution.

        Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
        accepted, respectively.

        TODO
        ----
        * Move initialization of 'sigma' to setting the per-particle variables.
        * Generalize to use MTS inner integrator.

        Examples
        --------

        Create a GHMC integrator.

        >>> temperature = 298.0 * simtk.unit.kelvin
        >>> collision_rate = 91.0 / simtk.unit.picoseconds
        >>> timestep = 1.0 * simtk.unit.femtoseconds
        >>> integrator = GHMCIntegrator(temperature, collision_rate, timestep)

        References
        ----------
        Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
        http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

        """

        # Initialize constants.
        kT = kB * temperature
        gamma = collision_rate

        # Create a new custom integrator.
        super(GHMCIntegrator, self).__init__(timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("kT", kT)  # thermal energy
        self.addGlobalVariable("b", numpy.exp(-gamma * timestep))  # velocity mixing parameter
        self.addPerDofVariable("sigma", 0) # velocity standard deviation
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("vold", 0)  # old velocities
        self.addPerDofVariable("xold", 0)  # old positions
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("potential_initial", 0)  # initial potential energy
        self.addGlobalVariable("potential_old", 0)  # old potential energy
        self.addGlobalVariable("potential_new", 0)  # new potential energy
        self.addGlobalVariable("accept", 0)  # accept or reject
        self.addGlobalVariable("naccept", 0)  # number accepted
        self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials
        self.addPerDofVariable("x1", 0)  # position before application of constraints

        #
        # Initialization.
        #
        self.beginIfBlock("ntrials = 0")
        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.endBlock()

        #
        # Allow context updating here.
        #
        self.addUpdateContextState()

        self.addComputeGlobal("potential_initial", "energy")
        for step in range(nsteps):
            #
            # Velocity randomization
            #
            self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
            self.addConstrainVelocities()

            # Compute initial total energy
            self.addComputeSum("ke", "0.5*m*v*v")
            self.addComputeGlobal("potential_old", "energy")
            self.addComputeGlobal("Eold", "ke + potential_old")
            self.addComputePerDof("xold", "x")
            self.addComputePerDof("vold", "v")
            # Velocity Verlet step
            self.addComputePerDof("v", "v + 0.5*dt*f/m")
            self.addComputePerDof("x", "x + v*dt")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
            self.addConstrainVelocities()
            # Compute final total energy
            self.addComputeSum("ke", "0.5*m*v*v")
            self.addComputeGlobal("potential_new", "energy")
            self.addComputeGlobal("Enew", "ke + potential_new")
            # Accept/reject, ensuring rejection if energy is NaN
            self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
            self.beginIfBlock("accept != 1")
            self.addComputePerDof("x", "xold")
            self.addComputePerDof("v", "-vold")
            self.addComputeGlobal("potential_new", "potential_old")
            self.endBlock()
            #
            # Velocity randomization
            #
            self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
            self.addConstrainVelocities()
            #
            # Accumulate statistics.
            #
            self.addComputeGlobal("naccept", "naccept + accept")
            self.addComputeGlobal("ntrials", "ntrials + 1")




