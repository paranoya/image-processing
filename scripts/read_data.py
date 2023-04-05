#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
from time import time


def run(dataset):
    t0 = time()

    object_name = 'No data read'
    data = np.array([])
    
    # 2D images:

    if dataset == 20:
        object_name = '2D Gaussian noise'
        data = np.random.normal(size=(1000, 1000)) * 42 + 666

    if dataset == 21:
        object_name = 'CIG 335'
        hdu = fits.open('data/CIG_335.fits')
        data = hdu[0].data[3000:4000, 1500:2500].astype(np.float32)

    if dataset == 22:
        object_name = 'NGC2420'
        hdu = fits.open('data/uamA_0033.fits')
        data = hdu[0].data.astype(np.float32)

    if dataset == 23:
        object_name = 'HGC 44 slice'
        hdu = fits.open('data/hcg44_cube_R.fits')
        data = hdu[0].data[69].astype(np.float32) # to make sure it's converted to float

    # 3D datacubes:

    if dataset == 30:
        object_name = '3D Gaussian noise'
        data = np.random.normal(size=(200, 200, 200)) * 42

    if dataset == 31:
        object_name = 'HGC 44'
        hdu = fits.open('data/hcg44_cube_R.fits')
        data = hdu[0].data[:, 150:350, 350:650].astype(np.float32)

    if dataset == 32:
        object_name = 'Synthetic WSRT cube'
        hdu = fits.open('data/sofiawsrtcube.fits')
        data = hdu[0].data.astype(np.float32)  # to make sure it's converted to float

    print(f"Read dataset {dataset}: \"{object_name}\" {data.shape} ({time()-t0:.3g} s)")
    return object_name, data


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    