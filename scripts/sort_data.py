#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time


def run(x):
    t0 = time()

    argsorted_data = np.argsort(x)
    argsorted_data = argsorted_data[np.isfinite(x[argsorted_data])]
    n_valid = argsorted_data.size
    
    print(f'Sorted {n_valid} finite measurements in ascending order ({time()-t0:.3g} s)')
    return argsorted_data, n_valid


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    