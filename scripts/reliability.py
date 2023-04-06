#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from matplotlib import pyplot as plt
from matplotlib import colors


def run(test_stat, area, plots=True):
    t0 = time()

    '''
    z_Chernoff_min = np.logspace(-12, -.01, 101)
    n_Chernoff_min = 1 - 2*np.log(1/test_stat.size)/(z_Chernoff_min-1-np.log(z_Chernoff_min))
    '''
    z_Chernoff_max = 1 + np.logspace(-3, 3, 101)
    n_Chernoff_max = 1 - 2*np.log(1/test_stat.size)/(z_Chernoff_max-1-np.log(z_Chernoff_max))
    interp_z_Chernoff = 10**np.interp(area, n_Chernoff_max[::-1], np.log10(z_Chernoff_max)[::-1])

    sorted_t = np.sort((test_stat/interp_z_Chernoff)[test_stat > 0])
    n_candidates = sorted_t.size
    number_above_t = n_candidates - np.arange(n_candidates)

    t = sorted_t[n_candidates//2:]
    n_t = number_above_t[n_candidates//2:]
    slope = np.log(n_t[-1]/n_t[0]) / np.log(t[-1]/t[0])
    power_law = n_t[0] * pow((t/t[0]), slope)

    difference = power_law - n_t
    index_0 = np.argmin(difference)
    index_1 = np.argmax(difference)
    t = t[index_0:index_1]
    n_t = n_t[index_0:index_1]
    slope = np.log(n_t[-1]/n_t[0]) / np.log(t[-1]/t[0])

    Chernoff_threshold = t[-1]
    n_sources = np.count_nonzero(sorted_t > Chernoff_threshold)
    print(f'{n_sources} found above threshold {Chernoff_threshold:.3g}')

    '''
    t = sorted_t[n_candidates//2+index_0:]
    n_t = number_above_t[n_candidates//2+index_0:]
    power_law = n_t[0] * pow((t/t[0]), slope)
    #Chernoff_threshold = np.max(t[n_t <= power_law])
    '''

    t_src = sorted_t[-n_sources:]
    n_t = number_above_t[-n_sources:]
    slope_bg = np.min(np.log(n_t[1:]/n_t[0]) / np.log(t_src[1:]/t_src[0]))
    power_law_bg = n_t[0] * pow((t_src/t_src[0]), slope_bg)
    n_reliable = int(np.max(n_t-power_law_bg))
    reliable_threshold = sorted_t[-n_reliable]
    slope_src = np.nanmax(np.log(n_t[-n_reliable]/n_t[:-n_reliable]) / np.log(t_src[-n_reliable]/t_src[:-n_reliable]))
    power_law_src = n_t[-n_reliable] * pow((t_src/reliable_threshold), slope_src)

    contamination = power_law_bg/(power_law_src+power_law_bg)
    contamination[0] = 1  # (just in case it was not)
    reliability = 1 - np.interp(test_stat/interp_z_Chernoff, t_src, contamination)

    true_overdensity = reliability > 0
    n_sources = np.count_nonzero(true_overdensity)
    reliability_threshold = 1 - np.interp(reliable_threshold, t_src, contamination)
    print(f'{n_sources} with reliability > 0, {n_reliable} above {reliability_threshold} ({time()-t0:.3g} s)')
    
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

        ax.plot(t_src, power_law_bg, 'r:', label='estimated contamination')
        ax.plot(t_src, power_law_src, 'b:', label='estimated sources')
        ax.axvline(Chernoff_threshold, c='b', ls='--', label=f'threshold={Chernoff_threshold:.3g} ({n_sources} sources)')
        ax.axvline(reliable_threshold, c='b', ls='-', label=f'threshold={reliable_threshold:.3g} ({n_reliable} sources)')
        #ax.plot(t, n_t-power_law, 'b:', label='estimated sources')

        ax.set_ylim(.5, 2*n_candidates)
        ax.legend()


        ax = axes[1, 0]  # ----------------- new panel
        ax.set_ylabel('reliability')

        ax.axvline(Chernoff_threshold, c='b', ls='--', label=f'reliability > 0 ({n_sources} sources)')
        ax.axvline(reliable_threshold, c='b', ls='-', label=f'reliability > {reliability_threshold:.3g} ({n_reliable} sources)')
        ax.plot(t_src, 1-contamination, 'k-')
        ax.legend()


        ax.set_xlabel('normalised test statistic')
        ax.set_xscale('log')
        ax.set_xlim(.1*Chernoff_threshold, 2*sorted_t[-1])

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
        ax.plot(n_Chernoff_max, reliable_threshold*z_Chernoff_max, 'b-')
        sc = ax.scatter(area[true_overdensity], test_stat[true_overdensity], label=f'{n_sources} sources',
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
    