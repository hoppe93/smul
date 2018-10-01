#!/usr/bin/env python3
"""
Main smul interface

HOW TO USE SMUL
---------------
Before using smul, the module must be initialized with a call to 'initialize()'.
Once that is done, 'waitForSignal()' should be called on all but the root process.
The 'waitForSignal()' function will wait for and process input vectors (returning
the resulting images to the root process) until the vector [0.0] is received, at
which point 'waitForSignal()' returns. On the root process, the function 'evalLikeness()'
should be called every time the likeness value corresponding to a particular
vector 'v' is needed. The 'evalLikeness()' function returns as soon as the likeness
has been computed.

ROOT PROCESS:
    smul.initialize('mycnf.conf')
    ...
    while likeness > MAX_LIKENESS:
        v = GetNewVector()
        likeness = smul.evalLikeness(v)

    smul.evalLikeness(smul.END_VECTOR)

OTHER PROCESSES:
    smul.initialize()
    smul.waitForSignal()
"""

import numpy as np
import sys

import Initialize
import Likeness
import SMPI
from SmulException import SmulException

# Global variables
END_VECTOR = [0.0]

def abort(): SMPI.abort()

def evalLikeness(v):
    """
    Compute the likeness of the image resulting from multiplying
    the Green's function with the distribution function generated from
    the input vector 'v' to the input image.
    NOTE: This function should (can) only be called from the root MPI process!

    v: Vector of values specifying how to generate the distribution function.
    """
    global END_VECTOR

    # Make sure only the root process can call us
    if not SMPI.is_root():
        raise SmulException("Only the root process may compute the likeness value.")

    # Distribute input vector
    n = SMPI.nproc()
    for i in range(1, n):
        SMPI.send(v, i, SMPI.TAG_INPUT_VECTOR)

    if v == END_VECTOR:
        return

    # Do multiplication
    I = smul_do(Initialize.distribution, Initialize.green, v)

    # Retrieve partial images
    for i in range(1, n):
        I += SMPI.recv(i, SMPI.TAG_IMAGE)

    # Evaluate likeness
    likeness = Likeness.compare(I, Initialize.realImage)

    return likeness

def exit():
    global END_VECTOR
    evalLikeness(END_VECTOR)

def getDfParameters():
    """
    Wait for distribution function parameters to
    be sent from the root MPI process
    """
    return SMPI.recv(SMPI.ROOT_PROC, SMPI.TAG_INPUT_VECTOR)

def initialize(config=""):
    """
    Initialize smul with the given configuration file.

    config: Name of file to read configuration from.
            (need not be provided to processes other
            than the root process)
    """
    SMPI.init()
    rank = SMPI.rank()

    Initialize.initialize(config)

def smul_do(df, gf, v):
    """
    Multiply the given Green's function with the given
    distribution function.

    df: DistributionFunction
    gf: GreensFunction
    v:  Vector of parameters specifying distribution function shape
    """
    return gf.multiply(df, v)

def waitForSignal():
    """
    Wait for, and process any incoming, vectors sent
    from the root MPI process. This function should
    be called from all processes but the root MPI
    process.

    This function blocks and reads multiple vectors
    until the 'END_VECTOR' is received.
    """
    global END_VECTOR

    v = getDfParameters()
    while v != END_VECTOR:
        # Evaluate image
        I = smul_do(Initialize.distribution, Initialize.green, v)

        # Send image to root process
        SMPI.send(data=I, dest=SMPI.ROOT_PROC, tag=SMPI.TAG_IMAGE)

        # Wait for next vector
        v = getDfParameters()


def main(argv):
    """
    Basic main function for testing purposes.

    argv: Command-line arguments.
    """
    if len(argv) != 1:
        print("ERROR: No configuration file provided.")
        sys.exit(-1)

    try:
        initialize(argv[0])

        if SMPI.rank() == SMPI.ROOT_PROC:
            evalLikeness([1.0])
            exit()
        else:
            waitForSignal()
    except Exception as ex:
        print("ERROR: {0}".format(ex))
        abort()


if __name__ == "__main__":
    main(sys.argv[1:])

