# An extremely simple distribution function that
# is one everywhere.

from DistributionFunction import DistributionFunction
import numpy as np

class UnitDistributionFunction(DistributionFunction):
    
    def __init__(self, nr, greenRadialGrid):
        super().__init__(nr, greenRadialGrid)

    def Eval(self, r, ppar, pperp, v, gamma=None, p2=None, p=None, xi=None):
        f = np.zeros(r.shape)
        f[np.where(ppar < 4)] = 1
        return f

