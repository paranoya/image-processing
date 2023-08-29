#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from scipy import ndimage


def run(data):
    """Separate diffuse emission from compact sources"""
    t0 = time()
    
    previous_background = data.copy()
    #previous_background[~np.isfinite(previous_background)] = data_offset
    data_valleys = None
    bg_fluctuations = np.inf
    residual_above_bg = 0

    while bg_fluctuations > residual_above_bg:
        # Find local minima
        valleys = np.where((previous_background[1:-1] < previous_background[:-2]) & (previous_background[1:-1] < previous_background[2:]))[0]
        smoothing_scale = 2 * data.size / valleys.size
        if data_valleys is None:
            data_valleys = valleys
            compact_scale = smoothing_scale
        #print(f'{valleys.size} valleys => smoothing_scale = {smoothing_scale:.1f} pixels')

        # Quantify fluctuations (previous background)
        waves = previous_background[1:-1][valleys[1:]] - previous_background[1:-1][valleys[:-1]]
        bg_fluctuations = np.sqrt(np.mean(waves**2))

        # Interpolation between minima
        weight = np.zeros_like(data)
        weight[1:-1][valleys] = 1
        weight = ndimage.gaussian_filter(weight, smoothing_scale)

        background = np.zeros_like(data)
        background[1:-1][valleys] = previous_background[1:-1][valleys]
        background = ndimage.gaussian_filter(background, smoothing_scale)/weight #/ np.where(weight > 0, weight, 1)
        background = np.fmin(previous_background, background)
        previous_background = background

        # Residuals (original minima vs new background)
        residual = data[1:-1][data_valleys] - background[1:-1][data_valleys]
        residual_above_bg = np.nanmedian(np.abs(residual))
        #print(f'   median absolute residual = {residual_above_bg:.3e}, bg_fluctuations = {bg_fluctuations:.3e}')

    print(f"mean/median background = {np.nanmean(background):.3e}/{np.nanmedian(background):.3e},  residual = {np.nanmean(data-background):.3e}/{np.nanmedian(data-background):.3e} ({time()-t0:.3g} s)")
    return compact_scale, smoothing_scale, background, residual_above_bg


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    