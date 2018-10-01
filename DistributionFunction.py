# Abstract base class for distribution functions

from abc import ABC, abstractmethod

class DistributionFunction(ABC):
    
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
    
