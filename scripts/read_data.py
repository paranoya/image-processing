#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from time import time


def run(dataset, section=None):
    t0 = time()

    object_name = 'No data read'
    data = np.array([])
    
    # 2D images:

    if dataset == 20:
        object_name = '2D Gaussian noise'
        data = np.random.normal(size=(1000, 1000)) * 42 + 666
        wcs = None

    if dataset == 21:
        object_name = 'CIG 335'
        hdu = fits.open('data/CIG_335.fits')
        data = hdu[0].data[3000:4000, 1500:2500].astype(np.float32)
        wcs = WCS(hdu[0].header)

    if dataset == 22:
        object_name = 'NGC2420'
        hdu = fits.open('data/uamA_0033.fits')
        data = hdu[0].data[:400,:400].astype(np.float32)
        #data = hdu[0].data.astype(np.float32)
        wcs = WCS(hdu[0].header)

    if dataset == 23:
        object_name = 'HGC 44 slice'
        hdu = fits.open('data/hcg44_cube_R.fits')
        data = hdu[0].data[69].astype(np.float32) # to make sure it's converted to float
        wcs = WCS(hdu[0].header).celestial

    # 3D datacubes:

    if dataset == 30:
        object_name = '3D Gaussian noise'
        data = np.random.normal(size=(200, 200, 200)) * 42
        wcs = None

    if dataset == 31:
        object_name = 'HGC 44'
        hdu = fits.open('data/hcg44_cube_R.fits')
        data = hdu[0].data[:, 150:350, 350:650].astype(np.float32)
        wcs = WCS(hdu[0].header).celestial

    if dataset == 32:
        object_name = 'Synthetic WSRT cube'
        hdu = fits.open('data/sofiawsrtcube.fits')
        data = hdu[0].data.astype(np.float32)  # to make sure it's converted to float
        wcs = WCS(hdu[0].header).celestial

    if dataset == 33:
        object_name = 'SoFiA test datacube'
        hdu = fits.open('data/sofia_test_datacube.fits')
        data = hdu[0].data.astype(np.float32)  # to make sure it's converted to float
        wcs = WCS(hdu[0].header).celestial

    if dataset == 34:
        # TODO: Make sure section is between (0, 0, 0) and (7, 6, 6)
        object_name = f"Section {section} in Tobias' noiseless datacube"
        hdu = fits.open('data/model_cube_blank_convol.fits')
        i = 175*section[0]
        j = max(0, 200*section[1] - 10)
        k = max(0, 200*section[2] - 10)
        #print(i,j,k)
        data = hdu[0].data[i:i+175, j:j+190, k:k+190].astype(np.float32)  # to make sure it's converted to float
        wcs = WCS(hdu[0].header).celestial

    if dataset == 35:
        # TODO: Make sure section is between (0, 0, 0) and (7, 6, 6)
        object_name = f"Section {section} in Tobias' synthetic datacube"
        hdu = fits.open('data/model_cube_noise_convol.fits')
        i = 175*section[0]
        j = max(0, 200*section[1] - 10)
        k = max(0, 200*section[2] - 10)
        #print(i,j,k)
        data = hdu[0].data[i:i+175, j:j+190, k:k+190].astype(np.float32)  # to make sure it's converted to float
        wcs = WCS(hdu[0].header).celestial

    print(f"Read dataset {dataset}: \"{object_name}\" {data.shape} ({time()-t0:.3g} s)")
    return object_name, data, wcs


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    