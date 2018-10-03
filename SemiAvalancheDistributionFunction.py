"""
Implementation of a semi-analytical avalanche distribution
of the form

    f(p, xi) = f0 * fp(p) * fxi(p,xi)

where
    
    p  = momentum
    xi = cosine of pitch angle
    g  = sqrt(1 + p^2) = Relativistic factor

    a  = parameter  (pronounced "alpha")
    A  = parameter
    f0 = parameter
    g0 = parameter

    fp(p)     = 1/(Gamma(a)*g0^a) * g^(a-1) * exp(-g/g0)

    fxi(p,xi) = A / (2*sinh(A)) * exp(A*xi)
"""

from DistributionFunction import DistributionFunction
import numpy.matlib
import numpy as np
import scipy.special

np.seterr(divide='ignore', invalid='ignore')

class SemiAvalancheDistributionFunction(DistributionFunction):
    
    def Eval(self, r, ppar, pperp, v, gamma=None, p2=None, p=None, xi=None):
        """
        Evaluate the avalanche distribution function defined by the
        vector 'v' in the point(s) given by (r, ppar, pperp).

        v should have the layout
          [a0,a1,...,an,A0,A1,...,An,f00,f01,...,f0n,g00,g01,...,g0n]
        where the index corresponds to the given radii.
        """
        V = self.PreprocessInputVector(v, len(r), nparams=4)

        a  = V[0,:]
        C  = V[1,:]
        f0 = V[2,:]
        g0 = V[3,:]

        # Compute other momentum quantities (if necessary)
        if p2 is None:    p2    = ppar**2 + pperp**2
        if p is None:     p     = np.sqrt(p2)
        if gamma is None: gamma = np.sqrt(1 + p2)
        if xi is None:    xi    = ppar / p

        Gamma = scipy.special.gamma(a)
        A = C*p*p / gamma

        fp  = 1/(Gamma*np.power(g0,a)) * np.power(gamma,a-1.0) * np.exp(-gamma/g0)
        fxi = A/(2.0*np.sinh(A)) * np.exp(A*xi)

        f = f0 * fp * fxi

        return f


########################
# Unit test
########################
def test():
    import matplotlib.pyplot as plt

    ad = SemiAvalancheDistributionFunction()

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
    a = 1.0
    C = 1.0
    f0 = 1.0
    g0 = 3.0

    v = np.array([a,C,f0,g0])

    f = ad.Eval(R, PPARl, PPERPl, v)
    F = np.reshape(f, (npperp, nppar))

    lf = np.log10(F)
    r = np.linspace(-10, -1, 10)
    plt.contourf(PPAR, PPERP, lf, r)
    plt.show()

if __name__ == '__main__':
    test()

