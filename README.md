# SOFT multiplier
The *SOFT multiplier* ``smul`` loads one or more
[SOFT](https://github.com/hoppe93/SOFT) Green's functions and can
be used to evaluate the images corresponding to a particular runaway electron
distribution function. The program utilizes MPI to load large, distributed
Green's function across multiple computer nodes.

## Example
``smul`` is intended to be used as a library for integration into other tools,
and not to be run on its own. An example program that utilizes ``smul`` could
be
```python
import smul
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialize using config file 'mycnf.conf'
smul.initialize('mycnf.conf')

if rank == 0:
    # Loop until desired likeness has been achieved
    MAX_LIKENESS = 1
    likeness = 100
    while likeness > MAX_LIKENESS
        v = GetNewVector()
        likeness = smul.evalLikeness(v)

    # Release other MPI processes
    # (make 'waitForSignal()' return)
    smul.exit()
else:
    smul.waitForSignal()
```
An example configuration file that could accompany this program is
```
[general]
# Path to Green's functions
green = green/function#d.mat

# Path to image to compare to (in MAT format)
image = myimg.mat

# Name of section defining distribution function
distribution = unitdistribution

[unitdistribution]
type = unit
```

## Green's functions
``smul`` takes [SOFT](https://github.com/hoppe93/SOFT) Green's functions as
input. The Green's functions must have the format ``r12ij``, and the size of
the image in the Green's function must be the same as in the image that it is
compared to.

When loading the Green's function, any occurence of ``#d`` in the filename will
be replaced by the index of the MPI process loading the function. Thus, very
large Green's functions can be split into several files and loaded separately
by individual MPI processes.

## Distribution functions
There are currently two types of distribution functions available in ``smul``.
These are

Name       | type | Description
-----------|------|--------------------------------------------
Avalanche  | unit | Analytical avalanche distribution function
Unit       | aava | A distribution that is one everywhere

The parametrization used for the distribution function is given when it is
evaluated. Parametrizations should be of the form

Name       | Parametrization
-----------|---------------------------------------------------
Avalanche  | ``[[a0,a1,...,an],[b0,b1,...,bn],[c0,c1,...,cn]]``
Unit       | N/A

## smul functions
Function        | Description
----------------|-----------------------------------------------------------------------
abort()         | Abort execution and close all MPI processes
evalLikeness(v) | Evaluate likeness of image resulting from vector ``v`` to input image
exit()          | Make all ``waitForSignal()`` functions return
initialize(c)   | Load the configuration file specified by ``c`` and prepare the run
waitForSignal() | Wait and respond to any vectors sent from root process

