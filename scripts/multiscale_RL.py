#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from scipy import ndimage


def run(data, smoothing_radii, resolution_boost=4, residual_accuracy=.1, max_iter=100):
    """Multiscale Richardson-Lucy deconvolution"""
    t0 = time()

    # remove NaN and negative values
    offset = np.nanmin(data)
    boosted_data = np.where(np.isfinite(data), data-offset, 0)

    # boost resolution
    original_pixel = np.arange(data.size * resolution_boost) / resolution_boost
    boosted_data = np.interp(original_pixel, np.arange(data.size), boosted_data)
    boosted_data = ndimage.gaussian_filter(boosted_data, resolution_boost/2)
    
    # main loop:
    mRL = np.ones(smoothing_radii.shape + boosted_data.shape) # initial guess
    epsilon = 1e-3*np.min(data[data > 0])  # to prevent underflow (0/0 division)
    old_rms_residual = 0
    rms_residual = np.min(data[data > 0])
    n_iter = 0
    while np.abs(old_rms_residual - rms_residual) > residual_accuracy*rms_residual and n_iter < max_iter:
        n_iter += 1
        old_rms_residual = rms_residual

        estimate = np.empty_like(mRL)
        for i, radius in enumerate(smoothing_radii):
            estimate[i] = ndimage.gaussian_filter(mRL[i], radius)
        estimate = np.sum(estimate, axis=0)
        rms_residual = np.std(boosted_data - estimate)

        for i, radius in enumerate(smoothing_radii):
             mRL[i] *= ndimage.gaussian_filter((boosted_data+epsilon) / (estimate+epsilon), radius)
        print(f'iteration {n_iter}/{max_iter}: rms_residual = {rms_residual:.2e} ({100*(rms_residual-old_rms_residual)/rms_residual:+.2f}%)')

    print(f"Multiscale Richardson-Lucy deconvolution ({time()-t0:.3g} s)")
    return boosted_data, mRL


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    