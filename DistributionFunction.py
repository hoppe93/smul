# Abstract base class for distribution functions

from abc import ABC, abstractmethod
import numpy as np
import smutil
import SMPI

class DistributionFunction(ABC):

    def __init__(self, nr, rmin, rmax, greenRadialGrid):
        #rmin = np.amin(greenRadialGrid)
        #rmax = np.amax(greenRadialGrid)
        self.radialGrid      = np.linspace(rmin, rmax, nr)
        self.greenRadialGrid = greenRadialGrid
    
    @abstractmethod
    def Eval(self, r, ppar, pperp, v, gamma=None, p2=None, p=None, xi=None):
        """
        Evaluate the given distribution function in the point(s)
        (r, ppar, pperp). Use the vector 'v' to specify the
        parameters of this distribution function.

        r:     Radial point(s) to evaluate the distribution function in.
        ppar:  Parallel momentum point(s) to evaluate the distribution function in.
        pperp: Perpendicular momentum point(s) to evaluate the distribution function in.
        v:     Vector of parameters (specific to each distribution function type)

        (Optional arguments)
        gamma: Pre-computed vector corresponding to sqrt(1 + ppar**2 + pperp**2)
        p2:    Pre-computed vector corresponding to ppar**2 + pperp**2
        p:     Pre-computed vector corresponding to sqrt(ppar**2 + pperp**2)
        xi:    Pre-computed vector corresponding to ppar/sqrt(ppar**2 + pperp**2)

        NOTE 1: Momentum is given units of mc (electron mass times
                the speed of light)
        NOTE 2: The length of r, ppar and pperp must be the same, while
                the length of v must be len(r)*number-of-parameters
        """
        while False:
            yield None
    
    def PreprocessInputVector(self, v, n, nparams):
        """
        Pre-process the input vector to give it a shape appropriate
        for generating the distribution

        v:    Input vector to reshape
        n:    Number of coordinates in grid
        npar: Number of parameters in model
        """
        l = v.size
        if l % nparams is not 0:
            smutil.error("AvalancheDistributionFunction: Input vector has invalid format: length is not a multiple of "+str(nparams)+" (number of parameters in model).")

        # Number of radial points in interface grid
        NR = self.radialGrid.size

        # Number of radial points in internal grid
        nr = self.greenRadialGrid.size
        nv = int(n/nr)

        abc = np.reshape(v, (nparams, NR))

        # Interpolate onto Green's function's radial grid
        a = np.interp(self.greenRadialGrid, self.radialGrid, abc[0,:])
        b = np.interp(self.greenRadialGrid, self.radialGrid, abc[1,:])
        c = np.interp(self.greenRadialGrid, self.radialGrid, abc[2,:])

        a = np.matlib.repmat(a, nv, 1).T
        b = np.matlib.repmat(b, nv, 1).T
        c = np.matlib.repmat(c, nv, 1).T

        a = np.reshape(a, (1,nv*nr))
        b = np.reshape(b, (1,nv*nr))
        c = np.reshape(c, (1,nv*nr))

        abc = np.array([a,b,c])

        return abc

