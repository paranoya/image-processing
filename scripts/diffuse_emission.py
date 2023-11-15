#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from scipy import ndimage


def find_minima(data, plateaus=False):
    """Return a boolean array with the positions of local minima (or plateaus)"""
    local_minimum = np.ones(data.shape, dtype=bool)
    m = local_minimum.ravel()
    d = data.ravel()
    for stride_bytes in data.strides:
        s = stride_bytes // data.itemsize
        if plateaus:
            m[:-s] &= (d[:-s] <= d[s:])
            m[s:] &= (d[s:] <= d[:-s])
        else:
            m[:-s] &= (d[:-s] <= d[s:])
            m[s:] &= (d[s:] <= d[:-s])
    return local_minimum


def find_scale(data):
    """Find characteristic separation between minima"""
    n_minima = np.count_nonzero(find_minima(data))
    n_valid = np.count_nonzero(np.isfinite(data))
    scale = np.power(n_valid / n_minima, 1/data.ndim)
    #radius = noise_scale / np.sqrt(8*np.log(2)) # FWHM -> Gaussian sigma
    return scale #/ np.sqrt(8*np.log(2)) # FWHM -> Gaussian sigma


def run(data, max_iter=100):
    """Separate diffuse emission from compact sources"""
    t0 = time()
    
    previous_background = data.copy()
    #inpaint = np.where(~ np.isfinite(previous_background))
    #print(inpaint[0].size)
    #previous_background[inpaint] = ndimage.gaussian_filter(previous_background, inpaint[0].size/2)[inpaint]
    data_minima = None
    bg_fluctuations = np.inf
    residual_above_bg = 0

    n_iter = 0
    while bg_fluctuations >= residual_above_bg and n_iter < max_iter:
        n_iter += 1
        
        # Find local minima
        weight = find_minima(previous_background).astype(float)
        weight[~np.isfinite(previous_background)] = 0
        minima = np.where(weight.ravel() > 0)
        smoothing_scale = np.power(data.size / np.count_nonzero(weight), 1/data.ndim)
        if data_minima is None:
            data_minima = minima
            compact_scale = smoothing_scale
        print(f'{minima[0].size} minima => smoothing_scale = {smoothing_scale:.1f} pixels')


        # Interpolation between minima
        weight = ndimage.gaussian_filter(weight, smoothing_scale)
        background = np.zeros_like(data)
        background.ravel()[minima] = previous_background.ravel()[minima]
        #inpaint = np.where(weight <= 0)
        #n_inpaint = inpaint[0].size
        #print('w', n_inpaint)
        #weight[inpaint] = 1 / n_inpaint
        #background[inpaint] = ndimage.gaussian_filter(background, n_inpaint/2)[inpaint]
        background = ndimage.gaussian_filter(background, smoothing_scale) / np.where(weight > 0, weight, np.nan)
        #previous_background[inpaint] = ndimage.gaussian_filter(previous_background, inpaint[0].size/2)[inpaint]


        # Quantify fluctuations (previous background)
        bg_fluctuations = np.nanstd(background - previous_background)
        background = np.fmin(previous_background, background)
        previous_background = np.fmin(previous_background, background)  # background

        # Residuals (original minima vs new background)
        residual = data.ravel()[data_minima] - background.ravel()[data_minima]
        residual_above_bg = np.nanmedian(np.abs(residual))
        #residual_above_bg = np.nanmedian(data-background)
        print(f'   median absolute residual = {residual_above_bg:.3e}, bg_fluctuations = {bg_fluctuations:.3e}')

    print(f"mean/median background = {np.nanmean(background):.3e}/{np.nanmedian(background):.3e},  residual = {np.nanmean(data-background):.3e}/{np.nanmedian(data-background):.3e} ({time()-t0:.3g} s)")
    return compact_scale, smoothing_scale, background, residual_above_bg


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    