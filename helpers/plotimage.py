#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt

f = h5py.File('realImage.mat', 'r')
plt.imshow(f['z'][:,:])
plt.show()

