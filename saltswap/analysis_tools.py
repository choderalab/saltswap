import numpy as np

"""
Functions to read and write saltswap simulation data
"""

def read_data(filename):
    """
    Read the number of salt molecules added, acceptance rates, and simulation run times for iterations of saltswap

    Parameters
    ----------
    filename: str
      the name of the file that contains the simulation data

    Returns
    -------
    data: numpy.ndarray
      array containing number of waters, number of salt pairs, acceptance probability, and run-time per iteration
    """
    filelines = open(filename).readlines()
    Nwats = []
    Nsalt = []
    Accprob = []
    time = []
    # i=3
    # step = int(filelines[i][0:5].strip())
    # while i-3 == step:
    # while i > 0:
    for i in range(3, len(filelines) - 3):
        # It appears some of the files have a varying length. This exception will pick those up.
        try:
            dummy = int(filelines[i][6:10].strip())
        except ValueError:
            break
        Nwats.append(int(filelines[i][6:10].strip()))
        Nsalt.append(int(filelines[i][13:18].strip()))
        #print('nsalt', int(filelines[i][13:18].strip()))
        Accprob.append(float(filelines[i][19:24].strip()))
        #print('acc', filelines[i][19:24].strip())
        time.append(float(filelines[i][55:68].strip()))
        #print('time', filelines[i][40:60].strip())
    return np.vstack((np.array(Nwats), np.array(Nsalt), np.array(Accprob), np.array(time)))


def read_work(filename):
    """
    Function to read the work to add or remove salt in a saltswap simulation.

    Parameter
    ---------
    filename: str
      the name of the file containing the work for each attempt

    Returns
    -------
    work: numpy.ndarray
      array of work values
    """
    filelines = open(filename).readlines()
    work = []
    for i in range(2, len(filelines)):
        work += [float(wrk) for wrk in filelines[i].split()]
    return np.array(work)