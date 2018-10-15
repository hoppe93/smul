"""
Implementation of the analytical avalanche distribution function
given in [Embreus et al., JPP (2018)].

The distribution function is given by

  f(p, xi) = a*b/c * g/p^2 * exp[-g/c - a*g*(1 + xi)]

where
  
  a, b, c are free parameters
  g = sqrt(1 + p^2)
"""

import numpy as np
import numpy.matlib
from DistributionFunction import DistributionFunction

np.seterr(divide='ignore', invalid='ignore')

class AvalancheDistributionFunction(DistributionFunction):
    def __init__(self, nr, rmin, rmax, greenRadialGrid):
        super().__init__(nr, rmin, rmax, greenRadialGrid)

    def Eval(self, r, ppar, pperp, v, gamma=None, p=None, p2=None, xi=None):
        """
        Evaluate the avalanche distribution function defined by the
        vector 'v' in the point(s) given by (r, ppar, pperp).

        v should have the layout
          [a0,a1,...,an,b0,b1,...,bn,c0,c1,...,cn]
        where the index corresponds to the given radii.
        """
        V = self.PreprocessInputVector(v, r.size, nparams=3)

        a = V[0,:]
        b = V[1,:]
        c = V[2,:]

        if p2 is None:    p2    = ppar**2 + pperp**2
        if p is None:     p     = np.sqrt(p2)
        if gamma is None: gamma = np.sqrt(1 + p2)
        if xi is None:    xi    = ppar / p

        f = a*b/c * gamma / p**2 * np.exp(-gamma/c - a*gamma*(1 - xi))

        return f

########################
# Unit test
########################
def test():
    import matplotlib.pyplot as plt

    ad = AvalancheDistributionFunction()

    nppar = 100
    npperp = 80

    ppar = np.linspace(0, 100, nppar)
    pperp = np.linspace(0, 100, npperp)
    PPAR, PPERP = np.meshgrid(ppar, pperp)

    n = nppar*npperp
    PPARl = np.reshape(PPAR, (1,n))
    PPERPl = np.reshape(PPERP, (1,n))

    R = np.ones((1,n))

    # Evaluate
    E = 10.0
    Z = 5.0

    a = E / Z
    b = 1.0
    c = 17*np.sqrt(Z+5)
    v = np.array([a,b,c])

    f = ad.Eval(R, PPARl, PPERPl, v)
    F = np.reshape(f, (npperp, nppar))

    lf = np.log10(F)
    r = np.linspace(-10, -1, 10)
    plt.contourf(PPAR, PPERP, lf, r)
    plt.show()

if __name__ == '__main__':
    test()

