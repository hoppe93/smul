"""
GENERATE TEST IMAGE FROM GREEN'S FUNCTION
"""

import h5py
import numpy as np
import sys
sys.addpath('..')

from GreensFunction import GreensFunction
import Initialize
import SMPI
import smul

smul.initialize('genimg.conf')

if SMPI.is_root():
    gf = smul.getGreensFunction()
    nr = gf.getNR()

    # Generate input vector
    E = 10.0        # Electric field
    Z = 5.0         # Effective charge

    # Linearly decaying radial profile
    b = np.linspace(1, 0, nr)
    # Momentum space parameters remain the same at all radii
    a = (E / Z) * np.ones(b.shape)
    c = (17*np.sqrt(Z+5)) * np.ones(b.shape)

    # Combine to form input vector
    v = np.concatenate([a,b,c])

    # Call other MPI processes to generate image
    I = smul.generateImage(v)

    # Write image
    with h5py.File(Initialize.realImage, 'w') as f:
        dset = f.create_dataset("z", I.shape, dtype='f')

    smul.exit()
else:
    smul.waitForSignal()

