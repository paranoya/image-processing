#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from matplotlib import pyplot as plt
from matplotlib import colors


def run(test_stat, area, plots=True):
    '''Separate a *sorted* collection of *possitive* values in two populations'''
    t0 = time()

    '''
    z_Chernoff_min = np.logspace(-12, -.01, 101)
    n_Chernoff_min = 1 - 2*np.log(1/test_stat.size)/(z_Chernoff_min-1-np.log(z_Chernoff_min))
    '''
    z_Chernoff_max = 1 + np.logspace(-3, 3, 101)
    n_Chernoff_max = 1 - 2*np.log(1/test_stat.size)/(z_Chernoff_max-1-np.log(z_Chernoff_max))
    interp_z_Chernoff = 10**np.interp(area, n_Chernoff_max[::-1], np.log10(z_Chernoff_max)[::-1])

    sorted_x = np.sort((test_stat/interp_z_Chernoff)[test_stat > 0])

    fraction_above = 1 - np.arange(sorted_x.size)/sorted_x.size
    
    delta = np.sqrt(np.nanmean(sorted_x**2)) / np.nanmean(sorted_x)
    index_0 = np.searchsorted(sorted_x, sorted_x[0]*delta)
    index_1 = np.searchsorted(sorted_x, sorted_x[-1]/delta)
    
    slope_left = -np.log(np.interp(sorted_x[index_0:index_1]/delta, sorted_x, fraction_above)/fraction_above[index_0:index_1]) /  np.log(delta)
    slope_right = np.log(np.interp(sorted_x[index_0:index_1]*delta, sorted_x, fraction_above)/fraction_above[index_0:index_1]) /  np.log(delta)
    #print(f'> DEBUG:\n  {delta, sorted_x[0]*delta, index_0, index_1}\n  {slope_left}\n  {slope_right}')

    index_cut = np.argmax(slope_right-slope_left)
    cut = sorted_x[index_0+index_cut]
    fraction_above_cut = fraction_above[index_0+index_cut]
    
    src_density = slope_right[index_cut] * fraction_above_cut*np.power(sorted_x/cut, slope_right[index_cut] - 1)
    bg_density = slope_left[index_cut] * fraction_above_cut*np.power(sorted_x/cut, slope_left[index_cut] - 1)
    probability_src = src_density / (src_density + bg_density)
    
    index_min = np.searchsorted(sorted_x, cut/delta)
    n_sources_above = np.cumsum(probability_src[index_min:])
    n_sources_above = n_sources_above[-1] + 1 - n_sources_above
    n_sources = max(1, int(np.sum(probability_src[index_min:])))
    threshold = sorted_x[-n_sources]
    #print(f'> DEBUG:  {np.sum(probability_src), n_sources}\n  {n_sources_above}')
    
    reliability = np.zeros_like(test_stat)
    sources = np.where(test_stat >= threshold * interp_z_Chernoff)
    #print(f'> DEBUG: {sources[0].size, np.count_nonzero(sorted_x > threshold), np.count_nonzero(test_stat/interp_z_Chernoff > threshold)}')
    reliability[sources] = np.interp(test_stat[sources]/interp_z_Chernoff[sources], sorted_x, probability_src)

    print(f"{n_sources} sources ({time()-t0:.3g} s)")

    if plots:
        fig_name = 'find_cut'
        plt.close(fig_name)
        fig = plt.figure(fig_name, figsize=(6, 6))
        axes = fig.subplots(nrows=2, ncols=1, squeeze=False,
                            sharex=True, sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})

        ax = axes[0, 0]  # ----------------- new panel
        #ax.set_title(object_name)
        ax.set_ylabel('number above threshold')
        ax.set_yscale('log')

        ax.plot(sorted_x, sorted_x.size*fraction_above, 'k-')
        ax.plot(sorted_x[index_min:], n_sources_above, 'b:')
        ax.plot(sorted_x[index_min:], sorted_x.size*fraction_above[index_min:]-n_sources_above, 'r:')

        ax.plot(sorted_x[index_min:], sorted_x.size*fraction_above_cut*np.power(sorted_x[index_min:]/cut, slope_left[index_cut]), 'r--')
        ax.plot(sorted_x[index_min:], sorted_x.size*fraction_above_cut*np.power(sorted_x[index_min:]/cut, slope_right[index_cut]), 'b--')

        #ax.set_ylim(.5, 2*n_candidates)


        ax = axes[1, 0]  # ----------------- new panel
        ax.set_ylabel('reliability')

        ax.plot(sorted_x, probability_src, 'k-')


        ax.set_xlabel('normalised test statistic')
        ax.set_xscale('log')
        #ax.set_xlim(.1*Chernoff_threshold, 2*sorted_t[-1])

        for ax in axes.flatten():
            ax.tick_params(which='both', direction='in')
            ax.grid(alpha=.5)
            ax.axvspan(cut/delta, cut*delta, color='k', alpha=.1)
            ax.axvline(threshold, c='k', ls='--', label=f'threshold={threshold:.3g} ({n_sources} sources)')
            ax.legend()

        fig.set_tight_layout(True)
        plt.show()

    if plots:
        plt.close('catalogue_selection')
        fig = plt.figure('catalogue_selection', figsize=(6, 6))
        axes = fig.subplots(nrows=1, ncols=1, squeeze=False,
                            sharex=True, sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})

        ax = axes[0, 0]
        #ax.set_title(object_name)
        ax.set_ylabel('test statistic')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('number of pixels')

        h2d = ax.hist2d(area, test_stat,
                        bins=[np.logspace(0.2, np.log10(np.max(area)), 30),
                              np.logspace(np.log10(1e-2*threshold), np.log10(1e3*threshold), 30)],
                        cmap='Greys', norm=colors.SymLogNorm(linthresh=1))
        ax.plot(n_Chernoff_max, threshold*z_Chernoff_max, 'b--')
        #ax.plot(n_Chernoff_max, reliable_threshold*z_Chernoff_max, 'b:')
        sc = ax.scatter(area[sources], test_stat[sources], label=f'{n_sources} sources',
                        marker='o', s=10, c=reliability[sources], cmap='nipy_spectral_r')
        ax.legend()
        cb = fig.colorbar(sc, ax=ax)
        cb.ax.set_ylabel('reliability')

        for ax in axes.flatten():
            ax.tick_params(which='both', direction='in')
        fig.set_tight_layout(True)
        plt.show()

    return reliability


def old_method(test_stat, area, plots=True):
    t0 = time()

    '''
    z_Chernoff_min = np.logspace(-12, -.01, 101)
    n_Chernoff_min = 1 - 2*np.log(1/test_stat.size)/(z_Chernoff_min-1-np.log(z_Chernoff_min))
    '''
    z_Chernoff_max = 1 + np.logspace(-3, 3, 101)
    n_Chernoff_max = 1 - 2*np.log(1/test_stat.size)/(z_Chernoff_max-1-np.log(z_Chernoff_max))
    interp_z_Chernoff = 10**np.interp(area, n_Chernoff_max[::-1], np.log10(z_Chernoff_max)[::-1])

    sorted_t = np.sort((test_stat/interp_z_Chernoff)[test_stat > 0])
    
    find_cut(sorted_t, plots)
    
    n_candidates = sorted_t.size
    number_above_t = n_candidates - np.arange(n_candidates)
    
    #t_mean = np.mean(sorted_t)
    #index_0 = np.searchsorted(sorted_t, t_mean)
    index_0 = 0
    index_1 = n_candidates - 1
    #slope_bg = np.log(number_above_t[index_1]/number_above_t[index_0]) / np.log(sorted_t[index_1]/sorted_t[index_0])
    #power_law_bg = number_above_t[index_0] * pow((sorted_t/sorted_t[index_0]), slope_bg)
    #while index_1 == n_candidates-1:
    while True:
        slope_bg = np.log(number_above_t[index_1]/number_above_t[index_0]) / np.log(sorted_t[index_1]/sorted_t[index_0])
        print(f'DEBUG: t[{index_0}, {index_1}] = ({sorted_t[index_0]}, {sorted_t[index_1]}), slope bg={slope_bg}')
        power_law_bg = number_above_t[index_0] * pow((sorted_t/sorted_t[index_0]), slope_bg)
        difference = number_above_t - power_law_bg
        index_max = index_0 + np.argmax(difference[index_0:])
        slope_max = np.log(number_above_t[index_1]/n_candidates) / np.log(sorted_t[index_1]/sorted_t[index_max])
        #power_law_max = n_candidates * pow((sorted_t/sorted_t[index_0]), slope_max)
        #difference = number_above_t - power_law_max
        if slope_max < slope_bg:
            index_0 = index_max
            index_1 = index_0 + 1 + np.argmin(difference[index_max+1:])
        else:
            break

    slope_src = np.log(number_above_t[index_1]/number_above_t[-1]) / np.log(sorted_t[index_1]/sorted_t[-1])
    power_law_src = number_above_t[-1] * pow((sorted_t/sorted_t[-1]), slope_src)
    difference = number_above_t - power_law_src
    index_2 = np.argmin(difference[index_1:])
    slope_min = np.log(number_above_t[index_1+index_2]/number_above_t[-1]) / np.log(sorted_t[index_1+index_2]/sorted_t[-1])
    power_law_min = number_above_t[-1] * pow((sorted_t/sorted_t[-1]), slope_min)
    print(f'DEBUG: index_1, 2 = {index_1, index_2}')
    if difference[index_1+index_2] < 0:
        slope_src = slope_min
        power_law_src = power_law_min
        n_reliable = int(power_law_min[index_1])
    else:
        n_reliable = number_above_t[index_1]
    reliable_threshold = sorted_t[-n_reliable]
    src_density = slope_src * number_above_t[-1]/sorted_t[-1] * pow((reliable_threshold/sorted_t[-1]), slope_src - 1)
    bg_density = slope_bg * number_above_t[index_0]/sorted_t[index_0] * pow((reliable_threshold/sorted_t[index_0]), slope_bg -1)
    reliability_threshold = src_density / (src_density + bg_density)
    print(f'DEBUG: slope bg={slope_bg}, src={slope_src}, n_reliable={n_reliable}')

    t = test_stat/interp_z_Chernoff
    good_t = t > 0
    t[~good_t] = sorted_t[0]/2
    src_density = slope_src * number_above_t[-1]/sorted_t[-1] * pow((t/sorted_t[-1]), slope_src - 1)
    bg_density = slope_bg * number_above_t[index_0]/sorted_t[index_0] * pow((t/sorted_t[index_0]), slope_bg -1)
    reliability = np.clip(src_density / (src_density + bg_density), a_min=0, a_max=1)
    

    Chernoff_threshold = sorted_t[index_1]
    true_overdensity = t > Chernoff_threshold
    n_sources = np.count_nonzero(true_overdensity)
    print(f'{n_sources} found above threshold {Chernoff_threshold:.3g}')
    reliability[~true_overdensity] = 0

    if plots:
        plt.close('test')
        fig = plt.figure('test', figsize=(6, 6))
        axes = fig.subplots(nrows=2, ncols=1, squeeze=False,
                            sharex=True, sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})

        ax = axes[0, 0]  # ----------------- new panel
        #ax.set_title(object_name)
        ax.set_ylabel('number above threshold')
        ax.set_yscale('log')

        ax.plot(sorted_t, number_above_t, 'k-')

        ax.plot(sorted_t[index_0:], power_law_bg[index_0:], 'r:')
        ax.plot(sorted_t[index_1:], power_law_src[index_1:], 'b--')
        ax.axvline(Chernoff_threshold, c='b', ls='--', label=f'threshold={Chernoff_threshold:.3g} ({n_sources} sources)')
        ax.axvline(reliable_threshold, c='b', ls=':', label=f'threshold={reliable_threshold:.3g} ({n_reliable} sources)')
        ax.axhline(n_reliable, c='b', ls=':')

        ax.set_ylim(.5, 2*n_candidates)
        ax.legend()


        ax = axes[1, 0]  # ----------------- new panel
        ax.set_ylabel('reliability')

        ax.axvline(Chernoff_threshold, c='b', ls='--')
        ax.axvline(reliable_threshold, c='b', ls=':', label=f'reliability threshold={reliability_threshold:.3g} ({n_reliable} sources)')
        ax.axhline(reliability_threshold, c='b', ls=':')
        src_density = slope_src * number_above_t[-1]/sorted_t[-1] * pow((sorted_t/sorted_t[-1]), slope_src - 1)
        bg_density = slope_bg * number_above_t[index_0]/sorted_t[index_0] * pow((sorted_t/sorted_t[index_0]), slope_bg -1)
        ax.plot(sorted_t, src_density / (src_density + bg_density), 'k-')
        ax.legend()


        ax.set_xlabel('normalised test statistic')
        ax.set_xscale('log')
        #ax.set_xlim(.1*Chernoff_threshold, 2*sorted_t[-1])

        for ax in axes.flatten():
            ax.tick_params(which='both', direction='in')
            ax.grid(alpha=.5)
        fig.set_tight_layout(True)
        plt.show()

    if plots:
        plt.close('catalogue_selection')
        fig = plt.figure('catalogue_selection', figsize=(6, 6))
        axes = fig.subplots(nrows=1, ncols=1, squeeze=False,
                            sharex=True, sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})

        ax = axes[0, 0]
        #ax.set_title(object_name)
        ax.set_ylabel('test statistic')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('number of pixels')

        h2d = ax.hist2d(area, test_stat,
                        bins=[np.logspace(0.2, np.log10(np.max(area)), 30),
                              np.logspace(np.log10(1e-2*Chernoff_threshold), np.log10(1e3*Chernoff_threshold), 30)],
                        cmap='Greys', norm=colors.SymLogNorm(linthresh=1))
        ax.plot(n_Chernoff_max, Chernoff_threshold*z_Chernoff_max, 'b--')
        ax.plot(n_Chernoff_max, reliable_threshold*z_Chernoff_max, 'b:')
        sc = ax.scatter(area[true_overdensity], test_stat[true_overdensity], label=f'{n_sources} sources ({n_reliable} reliable)',
                        marker='o', s=10, c=reliability[true_overdensity], cmap='nipy_spectral_r', vmax=reliability_threshold)
        ax.legend()
        cb = fig.colorbar(sc, ax=ax)
        cb.ax.set_ylabel('reliability')

        for ax in axes.flatten():
            ax.tick_params(which='both', direction='in')
        fig.set_tight_layout(True)
        plt.show()

    return reliability, reliability_threshold


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    