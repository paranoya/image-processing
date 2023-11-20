#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator

import numpy as np
from scipy import ndimage, special

#%% -----------------------------------------------------------------------------


def new_figure(fig_name, figsize=None, nrows=1, ncols=1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}):
    '''Automate the creation of figures'''
    
    plt.close(fig_name)
    if figsize is None:
        figsize = (10, 5*np.sqrt(nrows/ncols))
    fig = plt.figure(fig_name, figsize=figsize)
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                        sharex=sharex, sharey=sharey,
                        gridspec_kw=gridspec_kw
                       )
    fig.set_tight_layout(True)
    for ax in axes.flat:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        ax.tick_params(which='major', direction='inout', length=8, grid_alpha=.3)
        ax.tick_params(which='minor', direction='in', length=2, grid_alpha=.1)
        ax.grid(True, which='both')

    fig.suptitle(fig_name)
    
    return fig, axes


#%% -----------------------------------------------------------------------------


default_cmap = plt.get_cmap("gist_earth").copy()
default_cmap.set_bad('gray')


def colour_map(ax, cblabel, data, cmap=default_cmap, norm=None, xlabel=None, x=None, ylabel=None, y=None, projection_axis=None):

    if projection_axis is None:
        projection = data
    else:
        #projection = np.nanmean((data - np.nanmean(data, axis=0))**3, axis=0)
        #projection = np.nanmean(data, axis=projection_axis)
        projection = np.nanmax(data, axis=projection_axis)
    
    sigmas = np.linspace(-3, 3, 7)
    percentiles = 50 + 50 * special.erf(sigmas / np.sqrt(2))
    ticks = np.nanpercentile(data, percentiles)
    if norm is None:
        if ticks[-1] > 0:
            linthresh = np.median(data[data > 0])
            norm = colors.SymLogNorm(vmin=ticks[0], vmax=ticks[-1], linthresh=linthresh)
        else:
            norm = colors.Normalize(vmin=ticks[0], vmax=ticks[-1])

    if y is None:
        y = np.arange(projection.shape[0])
    if x is None:
        x = np.arange(projection.shape[1])

    im = ax.imshow(projection,
                   extent=(x[0]-(x[1]-x[0])/2, x[-1]+(x[-1]-x[-2])/2, y[0]-(y[1]-y[0])/2, y[-1]+(y[-1]-y[-2])/2),
                   interpolation='nearest', origin='lower',
                   cmap=cmap,
                   norm=norm,
                  )
    ax.set_aspect('auto')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    cb = plt.colorbar(im, ax=ax, orientation='vertical', shrink=.9)
    cb.ax.set_ylabel(cblabel)
    if ticks is not None:
        cb.ax.set_yticks(ticks=ticks, labels=[f'{value:.3g} ({percent:.1f}%)' for value, percent in zip(ticks, percentiles)])
    cb.ax.tick_params(labelsize='small')
    
    return im, cb, norm

#%% -----------------------------------------------------------------------------


def weighted_gaussian_filter(data, radius, weight=None):
    
    if weight is None:
        w = np.where(np.isfinite(data), 1., 0.)
    else:
        w = np.where(np.isfinite(weight) & np.isfinite(data), weight, 0.)

    #smooth_w = ndimage.gaussian_filter(w, radius, mode='constant')
    if np.sum(w) < data.size - .01:  # 1% of a pixel, just in case
        smooth_w = ndimage.gaussian_filter(w, radius, mode='constant')
    else:
        smooth_w = np.ones_like(data)
    
    return ndimage.gaussian_filter(np.where(w > 0, w*data, 0.), radius, mode='constant') / smooth_w

#%% -----------------------------------------------------------------------------


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    