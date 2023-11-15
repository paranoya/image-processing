#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from scipy import ndimage


def run(data, smoothing_radii, residual_accuracy=.01, max_iter=100, kernel_truncation=6):
    """Multiscale Richardson-Lucy deconvolution"""
    t0 = time()
    
    smoothing_radii = np.asarray(smoothing_radii)
    if len(smoothing_radii.shape) == 0:
        smoothing_radii = smoothing_radii.reshape((1,))

    # remove NaN and negative values
    offset = np.nanmin(data)
    boosted_data = np.where(np.isfinite(data), data-offset, np.nanmedian(data))
    
    # main loop:
    mRL = np.ones(smoothing_radii.shape + boosted_data.shape) # initial guess
    mRL *= boosted_data[np.newaxis, :]
    epsilon = 1e-3*np.min(data[data > 0])  # to prevent underflow (0/0 division)
    old_rms_residual = 0
    rms_residual = np.min(data[data > 0])
    n_iter = 0
    while np.abs(old_rms_residual - rms_residual) > residual_accuracy*rms_residual and n_iter < max_iter:
        n_iter += 1
        old_rms_residual = rms_residual

        estimate = np.empty_like(mRL)
        for i, radius in enumerate(smoothing_radii):
            estimate[i] = ndimage.gaussian_filter(mRL[i], radius, truncate=kernel_truncation)
        estimate = np.sum(estimate, axis=0)
        rms_residual = np.std(boosted_data - estimate)

        for i, radius in enumerate(smoothing_radii):
             mRL[i] *= ndimage.gaussian_filter((boosted_data+epsilon) / (estimate+epsilon), radius, truncate=kernel_truncation)
        print(f'iteration {n_iter}/{max_iter}: rms_residual = {rms_residual:.2e} ({100*(rms_residual-old_rms_residual)/rms_residual:+.2f}%)')
        
    print(f"Multiscale Richardson-Lucy deconvolution ({time()-t0:.3g} s)")
    return mRL


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    