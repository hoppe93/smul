# softmultiplier wrapper for MPI

from mpi4py import MPI
import numpy as np

_comm = None
_rank = None

ROOT_PROC = 0

# Tags
TAG_GREENSFUNCTION_NAME = 1
TAG_INPUT_VECTOR        = 2
TAG_IMAGE               = 3

def abort():
    global _comm
    _comm.Abort()

def init():
    global _comm, _rank
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()

def is_root():
    global _rank, ROOT_PROC
    return (_rank == ROOT_PROC)

def nproc():
    global _comm
    return _comm.Get_size()

def rank():
    global _rank
    return _rank

def recv(src, tag):
    global _comm
    data = _comm.recv(source=src, tag=tag)

    return data

def send(data, dest, tag):
    global _comm
    _comm.send(data, dest=dest, tag=tag)

