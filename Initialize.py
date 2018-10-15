"""
Initialize MPI and load up Green's functions

By: Mathias Hoppe, 2018
"""

import configparser
import os.path
import SMPI
import smutil
import scipy.io

from GreensFunction import GreensFunction
from AvalancheDistributionFunction import AvalancheDistributionFunction
from SemiAvalancheDistributionFunction import SemiAvalancheDistributionFunction
from UnitDistributionFunction import UnitDistributionFunction

from SmulException import SmulException

distribution = None
green = None
realImage = None
nr = None

# Global radial min/max
RMIN = None
RMAX = None

def constructDistributionFunction(name, config, rmin, rmax, greenRadialGrid):
    """
    Construct the distribution function to run with.

    name:   Name of distribution function to use.
    config: Configuration of the distribution.
    """
    global nr

    if 'nr' not in config:
        smutil.error("Number of radial points in interface grid not defined (nr).")
    else:
        nr = int(config['nr'])

    if config['type'] == 'avalanche':
        return AvalancheDistributionFunction(nr, rmin, rmax, greenRadialGrid)
    if config['type'] == 'semi':
        return SemiAvalancheDistributionFunction(nr, rmin, rmax, greenRadialGrid)
    if config['type'] == 'unit':
        return UnitDistributionFunction(nr, rmin, rmax, greenRadialGrid)
    else:
        smutil.error("Unrecognized distribution function type of '"+name+"': '"+config['type']+"'.")

def constructFilelist(basename):
    """
    Constructs a list of Green's function filenames that
    should be distributed across MPI processes.

    basename: Green's function basename (which contains one or
              more '#d' which are replaced with the corresponding
              MPI process IDs)
    """
    n = SMPI.nproc()
    filelist = []
    for i in range(0, n):
        f = basename.replace('#d', str(i))

        if not os.path.isfile(f):
            smutil.error("Green's function for process "+str(i)+" does not exist.")

        filelist.append(f)

    return filelist
            
def getGlobalRBounds():
    global RMIN, RMAX
    return RMIN, RMAX

def getNR():
    global nr
    return nr

def loadGreensFunction(filename):
    """
    Load the Green's function with the given name.

    filename: Name of Green's function to load.
    """
    return GreensFunction(filename)

def loadRealImage(filename):
    img = None

    # Try to load as older MAT file version
    try:
        matfile = scipy.io.loadmat(filename)

        img = matfile['z']
    # If it fails, try to load as HDF5
    except NotImplementedError:
        matfile = h5py.File(filename)

        img = matfile['z'][:,:]

    return img

def initialize(conf):
    """
    Initializes this process by reading the configuration
    file with name given by 'conf'.
    """
    global distribution, green, realImage, RMIN, RMAX

    print('Obtaining process rank')
    rank = SMPI.rank()

    # Load the configuration file
    print(str(rank)+': Loading configuration file')
    config = loadConfiguration(conf)

    fname = None
    if rank == 0:
        bname = config['general']['green']
        filelist = constructFilelist(bname)

        # Distribute filenames to processes (give 0 to this process)
        print('Distributing filenames to other processes')
        n = len(filelist)
        fname = filelist[0]
        for i in range(1, n):
            SMPI.send(filelist[i], i, SMPI.TAG_GREENSFUNCTION_NAME)

        if os.path.isfile(config['general']['image']):
            print('Loading real image...')
            realImage = loadRealImage(config['general']['image'])
        else:
            print('WARNING: Image to compare to did not exists. Assuming it will not be needed...')
            realImage = config['general']['image']
    else:
        # Get name of greens function
        fname = SMPI.recv(SMPI.ROOT_PROC, SMPI.TAG_GREENSFUNCTION_NAME)

    dfname = config['general']['distribution']
    print(str(rank)+": Loading Green's function...")
    green = loadGreensFunction(fname)
    rmin, rmax = green.getRadialBounds()

    # Distribute Green's function radial limits
    if rank == 0:
        RMIN = rmin
        RMAX = rmax

        # Get local limits
        n = SMPI.nproc()
        for i in range(1, n):
            mn, mx = SMPI.recv(i, SMPI.TAG_RADIAL_BOUNDS_LOCAL)
            if mn < RMIN: RMIN = mn
            if mx > RMAX: RMAX = mx

        # Distribute global limits
        for i in range(1, n):
            SMPI.send((RMIN, RMAX), i, SMPI.TAG_RADIAL_BOUNDS_GLOBAL)
    else:
        # Send local limits
        SMPI.send((rmin, rmax), SMPI.ROOT_PROC, SMPI.TAG_RADIAL_BOUNDS_LOCAL)
        # Get global limits
        RMIN, RMAX = SMPI.recv(SMPI.ROOT_PROC, SMPI.TAG_RADIAL_BOUNDS_GLOBAL)

    print(str(rank)+': Constructing distribution function')
    distribution = constructDistributionFunction(dfname, config[dfname], RMIN, RMAX, green.getSmallR())

def loadConfiguration(conf):
    """
    Load the configuration file with name 'conf'.

    conf: Name of configuration file.
    """
    config = configparser.ConfigParser()
    config.read(conf)

    # Validate contents
    if 'general' not in config:
        smutil.error("No general section in the configuration file.")
    if 'distribution' not in config['general']:
        smutil.error("No distribution function set in the configuration file.")
    else:
        dfname = config['general']['distribution']
        if dfname not in config:
            smutil.error("There is no section for the distribution function named '"+dfname+"' in configuration file.")
        else:
            if 'type' not in config[dfname]:
                smutil.error("The type of the distribution function '"+dfname+"' has not been specified.")

    # Verify format if Green's function name
    if 'green' not in config['general']:
        smutil.error("No filename provided for the Green's function.")
    if '#d' not in config['general']['green']:
        smutil.error("Green's function filename has invalid format. Expected '#d' to occur in name.")

    if 'image' not in config['general']:
        smutil.error("No truthful image provided.")

    return config

