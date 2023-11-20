#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from scipy import ndimage


def old(data, radius, residual_accuracy=.01, max_iter=100, kernel_truncation=6):
    """Richardson-Lucy deconvolution"""
    t0 = time()
    
    # remove NaN and negative values
    offset = np.nanmin(data)
    boosted_data = np.where(np.isfinite(data), data-offset, np.nanmedian(data)-offset)
    
    # main loop:
    RL = boosted_data.copy()
    epsilon = residual_accuracy*np.min(boosted_data[boosted_data > 0])  # to prevent underflow (0/0 division)
    model_residual = 0
    n_iter = 0
    converged = False
    while not converged:
        n_iter += 1
        model = np.empty_like(RL)
        model = ndimage.gaussian_filter(RL, radius, truncate=kernel_truncation)

        old_rms_residual = model_residual
        #rms_residual = np.std(boosted_data - model)
        #print(f'iteration {n_iter}/{max_iter}: rms_residual = {rms_residual:.2e} ({100*(rms_residual-old_rms_residual)/rms_residual:+.2f}%)')
        RL_residual = np.std(boosted_data - RL)
        model_residual = np.std(boosted_data - model)
        print(f'iteration {n_iter}/{max_iter}: rms residuals = {RL_residual:.2e} {model_residual:.2e}')
        print(f'   {np.std(model)}, {np.nanstd(boosted_data)}, {np.sqrt(np.nanvar(model) + np.var(boosted_data-model))}')
        if (np.abs(old_rms_residual - model_residual) > residual_accuracy*model_residual and
            n_iter < max_iter and
            RL_residual < model_residual):
            RL *= ndimage.gaussian_filter((boosted_data+epsilon) / (model+epsilon), radius, truncate=kernel_truncation)
        else:
            converged = True
        
    print(f"Richardson-Lucy deconvolution ({time()-t0:.3g} s)")
    return RL+offset, model+offset


def run(data, src_radius, bg_radius, residual_accuracy=.01, max_iter=100, kernel_truncation=6):
    """Richardson-Lucy deconvolution"""
    t0 = time()
    
    # remove NaN and negative values
    offset = np.nanmin(data)
    boosted_data = np.where(np.isfinite(data), data-offset, np.nanmedian(data)-offset)
    
    # main loop:
    src_RL = boosted_data.copy()
    bg_RL = np.nanmedian(src_RL) * np.ones_like(src_RL)
    epsilon = residual_accuracy*np.min(boosted_data[boosted_data > 0])  # to prevent underflow (0/0 division)
    model_residual = 0
    n_iter = 0
    converged = False
    while not converged:
        n_iter += 1
        src_model = ndimage.gaussian_filter(src_RL, src_radius, truncate=kernel_truncation)
        bg_model = ndimage.gaussian_filter(bg_RL, bg_radius, truncate=kernel_truncation)
        model = src_model + bg_model

        old_rms_residual = model_residual
        #rms_residual = np.std(boosted_data - model)
        #print(f'iteration {n_iter}/{max_iter}: rms_residual = {rms_residual:.2e} ({100*(rms_residual-old_rms_residual)/rms_residual:+.2f}%)')
        #RL_residual = np.std(boosted_data - RL)
        #model_residual = np.std(boosted_data - model)
        #print(f'iteration {n_iter}/{max_iter}: rms residuals = {RL_residual:.2e} {model_residual:.2e}')
        model_residual = np.std(boosted_data - src_model - bg_model)
        print(f'iteration {n_iter}/{max_iter}: rms residual = {model_residual:.2e}')
        if (np.abs(old_rms_residual - model_residual) > residual_accuracy*model_residual
            and n_iter < max_iter):
            #and RL_residual < model_residual):
            src_RL *= ndimage.gaussian_filter((boosted_data+epsilon) / (src_model+bg_model+epsilon), src_radius, truncate=kernel_truncation)
            bg_RL *= ndimage.gaussian_filter((boosted_data+epsilon) / (src_model+bg_model+epsilon), bg_radius, truncate=kernel_truncation)
            RL = src_RL + bg_RL
            #src_RL, bg_RL = np.where(RL > model, (src_RL+.5*bg_RL, .5*bg_RL), (.5*src_RL, .5*src_RL+bg_RL))
            src_RL, bg_RL = np.where(RL > model, (src_RL+bg_RL, 0*bg_RL), (0*src_RL, src_RL+bg_RL))
        else:
            converged = True
        
    print(f"Richardson-Lucy deconvolution ({time()-t0:.3g} s)")
    return src_RL, bg_RL


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    