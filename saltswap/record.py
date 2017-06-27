"""
File to record saltswap simulation data with netcdf.
"""
from netCDF4 import Dataset
import numpy as np
import simtk

# The simulation parameters that will be recorded from Swapper.
swapper_control_attributes = ('temperature', 'pressure', 'delta_chem', 'npert')

# Establishing default units for recording data. The netcdf file will also be storing units where appropriate.
variable_states = {'volume':simtk.unit.nanometer**3, 'potential energy':simtk.unit.kilojoule_per_mole}

# All variables that will be stored in the sampler state group
sample_variables = ['identities','species counts','cumulative work','proposal','naccepted']
for var in variable_states:
    sample_variables.append(var)

class CreateNetCDF(object):
    """
    Class to record data from MD and saltswap simulations
    """
    def __init__(self, filename):
        """
        Create a netcdf file to store saltswap simulation data.

        Parameters
        ----------
        filename: str
            the name of the netcdf file that will be created
        """
        self.ncfile = Dataset(filename, 'w', format='NETCDF4')
        self.scalar_dim = self.ncfile.createDimension('scalar', 1)
        self.string_dim = self.ncfile.createDimension('string', 0)
        self.ncfile.createDimension('iteration', None)
        self.ncfile.createDimension('attempt', None)

    def init_control_variables(self, swapper, variable_dic=None):
        """
        Initialize netcdf file to store simulation inputs, such as integrator timestep, temperature, etc.

        Parameters
        ----------
        swapper: saltswap.swapper
            the driver for saltswap
        variable_dic: dic
            dictionary containing additional parameters to be stored
        """
        # Creating a group for the simulation parameters, e.g. temperature, chemical potential, timesteps etc,
        control_parms_group = self.ncfile.createGroup('Control parameters')

        # Saving the control parameters that are accessible from the saltswap swapper:
        for attribute in swapper_control_attributes:
            atrb = getattr(swapper, attribute)
            if type(atrb) == simtk.unit.quantity.Quantity:
                var = control_parms_group.createVariable(attribute, 'f4')
                control_parms_group.variables[attribute][0] = atrb._value
                var.unit = atrb.unit._name
            elif type(atrb) == float:
                control_parms_group.createVariable(attribute, 'f4')
                control_parms_group.variables[attribute][0] = atrb
            elif type(atrb) == int:
                control_parms_group.createVariable(attribute, 'i4')
                control_parms_group.variables[attribute][0] = atrb

        # Saving the control parameters that listed in the parameter dictionary:
        for attribute in variable_dic:
            atrb = variable_dic[attribute]
            if type(atrb) == simtk.unit.quantity.Quantity:
                var = control_parms_group.createVariable(attribute, 'f4')
                control_parms_group.variables[attribute][0] = atrb._value
                var.unit = atrb.unit._name
            elif type(atrb) == float:
                control_parms_group.createVariable(attribute, 'f4')
                control_parms_group.variables[attribute][0] = atrb
            elif type(atrb) == int:
                control_parms_group.createVariable(attribute, 'i4')
                control_parms_group.variables[attribute][0] = atrb
            elif type(atrb) == str:
                control_parms_group.createVariable(attribute, str, 'string')
                control_parms_group.variables[attribute][:] = np.array([atrb], "O")

        self.ncfile.sync()

    def init_sample_state_variables(self, swapper, sams_max=None):
        """
        Recording the simulation variables.

        Parameters
        ----------
        swapper: saltswap.swapper
            the driver for saltswap
        sams_max: int or None.
            if running a Self Adjusted Mixture Simulation, this is maximum number of salt pairs that will be inserted.
        """
        sample_state_group = self.ncfile.createGroup('Sample state data')

        # Scalar quantities with their units
        for variable in variable_states:
            var = sample_state_group.createVariable(variable, 'f4', ('iteration'), zlib=True)
            var.unit = variable_states[variable].get_name()

        # Which molecule is which
        sample_state_group.createDimension('identity', len(swapper.mutable_residues))
        sample_state_group.createVariable('identities', 'i4', ('iteration', 'identity'), zlib=True)

        # How many water, cations, and anions there are.
        sample_state_group.createDimension('species', len(swapper.get_identity_counts()))
        sample_state_group.createVariable('species counts', 'i4', ('iteration', 'species'), zlib=True)

        # The cumulative work for each NCMC step in thermal units (i.e. unitless)
        sample_state_group.createDimension('attempt', None)
        sample_state_group.createDimension('npert', swapper.npert + 1)
        var = sample_state_group.createVariable('cumulative work', 'f4', ('iteration', 'attempt', 'npert'), zlib=True)
        var.unit = 'unitless'
        # The corresponding proposal for the NCMC protocol work
        sample_state_group.createDimension('proposal', 2)
        sample_state_group.createVariable('proposal', 'i4', ('iteration', 'attempt', 'proposal',), zlib=True)

        # The number of saltswap moves that have been accepted
        sample_state_group.createVariable('naccepted', 'i4', ('iteration', 'attempt',), zlib=True)
        sample_state_group.createVariable('nattempted', 'i4', ('iteration', 'attempt'), zlib=True)

        # The log acceptance probability for each attempt
        sample_state_group.createVariable('log_accept', 'f4', ('iteration', 'attempt'), zlib=True)

        # The biases from self-adjusted mixture sampling.
        if sams_max is not None:
            sample_state_group.createDimension('nsalt', sams_max + 1)
            sample_state_group.createVariable('sams bias', 'f8', ('iteration', 'nsalt'), zlib=True)

        self.ncfile.sync()

    def create_netcdf(self, swapper, variable_dic=None, sams_max=None):
        """
        Create a
        Parameters
        ----------
        swapper: saltswap.swapper
            the driver for saltswap
        variable_dic: dic
            dictionary containing additional parameters to be stored
        sams_max: int or None
            if running SAMS, the maximum number of salt pairs that will be inserted.
        """
        self.init_control_variables(swapper, variable_dic)
        self.init_sample_state_variables(swapper, sams_max)

        self.ncfile.sync()

        return self.ncfile

def record_netcdf(ncfile, context, swapper, iteration, attempt=0, sams_bias=None, sync=True):
    """
    Store the variables in the context and swapper objects.

    Parameters
    ----------
    ncfile: netCDF4._netCDF4.Dataset
        Netcdf data object that has already been initialized to record saltswap simulation data.
    context: simtk.openmm.openmm.Context
        Contains the OpenMM simulation data and parameters
    swapper: saltswap.swapper
        the driver for saltswap
    sams_bias: numpy.ndarray or None.
        the Self Adjusted Mixture Sampling weights.
    """
    # Openmm state information
    state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)

    volume = state.getPeriodicBoxVolume()
    var_units = variable_states['volume']
    ncfile.groups['Sample state data']['volume'][iteration] = volume.value_in_unit(var_units)

    potential_energy = state.getPotentialEnergy()
    var_units = variable_states['potential energy']
    ncfile.groups['Sample state data']['potential energy'][iteration] = potential_energy .value_in_unit(var_units)

    # saltswap state information
    ncfile.groups['Sample state data']['identities'][iteration, :] = swapper.stateVector
    ncfile.groups['Sample state data']['species counts'][iteration, :] = swapper.get_identity_counts()

    # saltswap MCMC information
    ncfile.groups['Sample state data']['proposal'][iteration, attempt, :] = swapper.proposal
    ncfile.groups['Sample state data']['cumulative work'][iteration, attempt, :] = swapper.cumulative_work
    ncfile.groups['Sample state data']['naccepted'][iteration, attempt] = swapper.naccepted
    ncfile.groups['Sample state data']['nattempted'][iteration, attempt] = swapper.nattempted
    ncfile.groups['Sample state data']['log_accept'][iteration, attempt] =  swapper.log_accept

    if sams_bias is not None:
        ncfile.groups['Sample state data']['sams bias'][iteration, :] = sams_bias

    if sync:
        ncfile.sync()

    return ncfile