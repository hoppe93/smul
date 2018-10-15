# An extremely simple distribution function that
# is one everywhere.

from DistributionFunction import DistributionFunction
import numpy as np

class UnitDistributionFunction(DistributionFunction):
    
    def __init__(self, nr, rmin, rmax, greenRadialGrid):
        super().__init__(nr, rmin, rmax, greenRadialGrid)

    def Eval(self, r, ppar, pperp, v, gamma=None, p2=None, p=None, xi=None):
        f = np.zeros(r.shape)
        f[np.where(ppar < 4)] = 1
        return f

