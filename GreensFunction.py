
import h5py
import numpy as np
import numpy.matlib

class GreensFunction:
    
    def __init__(self, filename):
        """
        Constructor
        """
        self.nr = None
        self.smallR = None
        self.R = None
        self.PPAR = None
        self.PPERP = None
        self.GAMMA = None
        self.XI = None
        self.P = None
        self.P2 = None
        self.FUNC = None

        self.loadHDF5(filename)

    def loadHDF5(self, filename):
        """
        Loads the Green's function file
        with the given name using h5py.
        """
        matfile = h5py.File(filename)
        
        # Make sure the file has the required fields
        fields = ['func', 'param1', 'param2', 'param1name', 'param2name', 'pixels', 'r', 'format']
        for field in fields:
            if field not in matfile:
                raise ValueError("Badly formatted Green's function. Missing field '"+field+"'")

        # Verify that images have been placed continuously
        #frmt = [x.decode() for x in matfile['format']]
        #frmt = matfile['format'][:,0]
        frmt = ''.join([str(chr(x)) for x in matfile['format'][:,0]])

        if not frmt.endswith('ij'):
            raise ValueError("Badly formatted Green's function. The Green's function format must end in 'ij'. Format: "+frmt)
        if frmt != 'r12ij':
            #    raise ValueError("The Green's function must have one spatial and two momentum dimensions.")
            #if len(frmt) is not 5:
            raise ValueError("Unrecognized Green's function format: "+frmt)

        self.NPIXELS = int(matfile['pixels'][0,0])
        self.FUNC = matfile['func'][:,:]
        #self.FUNC = matfile['func']

        # Generate phase-space
        p1 = matfile['param1']
        p2 = matfile['param2']
        p1name = ''.join([str(chr(x)) for x in matfile['param1name'][:,0]])
        p2name = ''.join([str(chr(x)) for x in matfile['param2name'][:,0]])
        ppar, pperp = self.toPparPperp(p1, p2, p1name, p2name)

        tr = matfile['r'][:,0]
        self.nr = tr.size
        n = self.nr * ppar.size

        self.FUNC = np.reshape(self.FUNC, (n, self.NPIXELS*self.NPIXELS)).T

        self.smallR = tr
        self.R = np.matlib.repmat(tr, 1, ppar.size)
        self.PPAR  = np.reshape((np.matlib.repmat(ppar,  1, self.nr)).T, (1, n))
        self.PPERP = np.reshape((np.matlib.repmat(pperp, 1, self.nr)).T, (1, n))

        self.P2    = self.PPAR**2 + self.PPERP**2
        self.P     = np.sqrt(self.P2)
        self.GAMMA = np.sqrt(1.0 + self.P2)
        self.XI    = self.PPAR / self.P

    def toPparPperp(self, p1, p2, p1name, p2name):
        """
        Takes in two momentum parameters (momentum 1 & 2)
        and returns the corresponding ppar/pperp grid
        (as a meshgrid).

        p1, p2:         Vectors momentum-space
                        parameter 1 & 2 values
        p1name, p2name: Names of momentum-space
                        parameter 1 & 2
        """
        ppar, pperp = None, None

        if p1name == 'ppar' and p2name == 'pperp':
            ppar, pperp = np.meshgrid(p1, p2)
        elif p1name == 'pperp' and p2name == 'ppar':
            ppar, pperp = np.meshgrid(p2, p1)
        elif p1name == 'p' and p2name == 'pitch':
            p, xi = np.meshgrid(p1, np.cos(p2))
            ppar = p * xi
            pperp = np.sqrt(p*p - ppar*ppar)
        elif p1name == 'pitch' and p2name == 'p':
            p, xi = np.meshgrid(p2, np.cos(p1))
            ppar = p * xi
            pperp = np.sqrt(p*p - ppar*ppar)
        else:
            raise ValueError("Unrecognized pair of momentum coordinates: '"+p1name+"' and '"+p2name+"'")

        return ppar, pperp

    def getFunction(self): return self.FUNC
    def getNR(self): return self.nr
    def getNpixels(self): return self.NPIXELS
    def getPhaseSpace(self): return self.R, self.PPAR, self.PPERP
    def getRadialBounds(self): return np.amin(self.smallR), np.amax(self.smallR)
    def getSmallR(self): return self.smallR

    def multiply(self, distributionFunction, v):
        """
        Multiply this Green's function with the
        given distribution function.

        distributionFunction: Function handle to function
                              that evaluates the distribution
                              function to evaluate with.
        v:                    Vector of parameters specifying shape
                              of distribution function. Shape:
                              [a0,a1,...,an,b0,b1,...,bn,c0,c1,...,cn]
                              where each index corresponds to an
                              individual radius.
        """
        gf = self.FUNC
        r, ppar, pperp = self.getPhaseSpace()
        npixels = self.NPIXELS
        npixels2 = npixels*npixels
        n = r.shape[1]
        I = np.zeros(npixels2)
        f = distributionFunction.Eval(r, ppar, pperp, v, gamma=self.GAMMA, p2=self.P2, p=self.P, xi=self.XI)
        
        I = np.matmul(gf, f.T)
        #for i in range(0, n):
        #    s = gf[(i*npixels2):((i+1)*npixels2)] * f[0,i]
        #    I += s[:,0]

        I = np.reshape(I, (npixels, npixels))
        return I

if __name__ == '__main__':
    from UnitDistributionFunction import UnitDistributionFunction
    import time
    import matplotlib.pyplot as plt

    gf = GreensFunction('examples/green.mat')
    df = UnitDistributionFunction()

    tic = time.time()
    I = gf.multiply(df, [1.0])
    toc = time.time()

    #plt.imshow(I)
    #plt.show()

    print('Execution time: %s' % (toc - tic))

