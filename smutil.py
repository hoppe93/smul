# Various utility functions

import SMPI

def error(msg):
    print('ERROR: '+msg)
    SMPI.abort()

