import logging
import matplotlib.pyplot as plt
import numpy as np
from . import utils
import scipy.stats
import time
import pandas as pd
import warnings
log = logging.getLogger(__name__)
colors = plt.get_cmap('tab10').colors


def estimate_threshold_by_Ntrials(epochs, N_trials_one_polarity=256*.125*np.arange(2,9), seed=0,
                                  second_filter='pre-average',
                                  second_filter_settings={'highpass': 300, 'lowpass': 3000, 'order': 1},
                                  plot_results=False, N_resamples=50, paper_plot=False, **kwargs):
    N_trials_one_polarity = N_trials_one_polarity.astype(int)
    Needed_N_per_polarity = 256

    # Drop empty trials
    all_values_0 = np.all(epochs == 0, axis=1)
    dropped_all = False
    if any(all_values_0):
        dropped_all = all(all_values_0)
        log.warning(f'\n {all_values_0.sum()}/{len(all_values_0)} epochs in this dataset have values of all 0.'
                    f' Dropping these epochs.')
        epochs = epochs[~all_values_0]

    levels = epochs.index.get_level_values('level').unique().values
    polarities = epochs.index.get_level_values('polarity').unique().values
    unV, unN = np.unique(np.diff(epochs.keys().values), return_counts=True)
    fs = 1 / unV[unN.argmax()]
    N_by_polarity = epochs.groupby(['polarity', 'level']).size()
    N_varies = len(N_by_polarity.unstack('polarity').min(axis=1).unique()) > 1
    N_by_polarity = epochs.groupby(['polarity', 'level']).size()
    if len(N_by_polarity) == 0:
        if dropped_all:
            N_min_global = 0
        else:
            raise RuntimeError('N_by_polarity is empty but not all epochs were dropped. If this happened its a bug.')
    else:
        N_min_global = N_by_polarity.min()
    if N_min_global < Needed_N_per_polarity:
        if dropped_all:
            msg = f'All epochs were dropped because they all had values of all zero.'
        else:
            msg = (f'There is at least one condition with only {N_min_global} reps. Minimum needed for SNR calcs is '
                   f'{Needed_N_per_polarity}. There are {len(epochs)} total reps for this recording')
        log.warning(msg)
        datapars = {'fs': fs, 'levels': levels}
        algpars = {'peak_lag_threshold': kwargs['peak_lag_threshold'],
                   'N_shuffles': kwargs['N_shuffles']}
        fit_results = {'threshold': np.nan, 'threshold_XCp_near_0': np.nan,
                       'fit_XCp_near_0': None, 'fit_KSs': None,
                       'peak_lag': np.nan, 'peak': np.nan, 'xc0': np.nan,
                       'xcLm': np.nan, 'xcLstd': np.nan,
                       'algpars': algpars, 'datapars': datapars}
        fit_results['threshold_XC0m'] = np.nan
        fit_results['threshold_XCp_near_0'] = np.nan
        fit_results['threshold_KSs'] = np.nan
        fit_results['fit_XCp_near_0'] = None
        fit_results['fit_KSs'] = None
        fit_results['fit_XC0m'] = None
        fit_results['comptime'] = np.nan
        fit_results['status'] = 'Skipped'
        fit_results['status_message'] = msg
        return fit_results, plt.figure()

    # filter data if requested
    if second_filter == 'pre-average':
        epochs[:] = utils.filter(epochs.values, fs, **second_filter_settings)

    fit_results, fig_handle = estimate_threshold(epochs, plot_results=True, second_filter=None, **kwargs)

    rn = np.random.RandomState(seed)
    thresholds = np.zeros((len(N_trials_one_polarity), N_resamples))
    # thresholds_by_resample_mean = np.zeros((len(N_trials_one_polarity), N_resamples))
    # thresholds_by_resample_std = np.zeros((len(N_trials_one_polarity), N_resamples))
    thresholds_by_resample = np.zeros((len(N_trials_one_polarity), N_resamples, kwargs['N_shuffles']))
    thresholds_by_resample_nans = np.full((len(N_trials_one_polarity), N_resamples), "", dtype=object)
    status_messages = np.full((len(N_trials_one_polarity), N_resamples), "", dtype=object)
    for i, N_trials_one_polarity_ in enumerate(N_trials_one_polarity[:-1]):
        print(i)
        if paper_plot and (i==0):
            all_masks = []
        for j in range(N_resamples):
            keep_ind_masks =  np.zeros(Needed_N_per_polarity, dtype=bool)
            keep_ind_masks[rn.choice(Needed_N_per_polarity, N_trials_one_polarity_, replace=False)] = True
            if paper_plot and (i==0):
                all_masks.append(keep_ind_masks)
            fit_results_, fig_handle_ = estimate_threshold(epochs, plot_results=plot_results,
                                                           keep_ind_masks=(keep_ind_masks, keep_ind_masks),
                                                           second_filter=None, **kwargs)
            if fit_results['status_message'] != fit_results_['status_message']:
                 warnings.warn(f"Status message mismtach. Was {fit_results['status_message']}, now {fit_results_['status_message']}")
            status_messages[i,j] = fit_results_['status_message']
            thresholds[i,j] = fit_results_['threshold']
            # thresholds_by_resample_mean[i,j] = fit_results_['threshold_mean']
            # thresholds_by_resample_std[i, j] = fit_results_['threshold_std']
            thresholds_by_resample[i,j,:] = fit_results_['thresholds']
            thresholds_by_resample_nans[i, j] = fit_results_['thresholds_nans']
    thresholds[-1,:] = fit_results['threshold']
    thresholds_by_resample[-1, :, :] = fit_results['thresholds']
    thresholds_by_resample[np.isinf(thresholds_by_resample) & (thresholds_by_resample > 0)] = levels.max() + 5
    thresholds_by_resample[np.isinf(thresholds_by_resample) & (thresholds_by_resample < 0)] = levels.min() - 5
    thresholds_by_resample_mean = np.nanmean(thresholds_by_resample, axis=2)
    thresholds_by_resample_std = np.nanstd(thresholds_by_resample,axis=2)
    status_messages[-1, :] = fit_results['status_message']
    fit_results['threshold'] = thresholds
    fit_results['threshold_mean'] = thresholds.mean(axis=1)
    fit_results['threshold_std'] = thresholds.std(axis=1)
    fit_results['thresholds_by_resample'] = np.round(thresholds_by_resample, 2)
    fit_results['thresholds_by_resample_mean'] = np.round(thresholds_by_resample_mean, 2)
    fit_results['thresholds_by_resample_std'] = np.round(thresholds_by_resample_std, 2)
    fit_results['N_trials_one_polarity'] = N_trials_one_polarity
    fit_results['status_messages'] = status_messages
    fit_results['all_status_messages_same'] = len(np.unique(status_messages)) == 1
    thresholds[np.isinf(thresholds) & (thresholds > 0)] = levels.max() + 5
    thresholds[np.isinf(thresholds) & (thresholds < 0)] = levels.min() - 5
    if paper_plot:
        order = thresholds_by_resample_std[0,:].argsort()
        ranks = order.argsort()
        imedian = np.where(ranks == round(N_resamples/2))[0][0]
        keep_ind_masks = all_masks[imedian]
        fit_results_lowN, fig_handle_lowN = estimate_threshold(epochs, plot_results=True,
                                                       keep_ind_masks=(keep_ind_masks, keep_ind_masks),
                                                       second_filter=None, **kwargs)
        f, ax = plt.subplots(1, figsize=(3, 5))
        xbins = np.hstack((N_trials_one_polarity,N_trials_one_polarity[-1]+np.diff(N_trials_one_polarity[:2])))-np.diff(N_trials_one_polarity[:2])/2
        ylims2 = [.8, 4]
        ybins2 = np.linspace(ylims2[0], ylims2[1], 25)

        h = ax.hist2d(np.tile(N_trials_one_polarity*2, (N_resamples, 1)).T.flatten(),
                       thresholds_by_resample_std.flatten(), bins=(xbins*2, ybins2), cmap='gray_r')
        h[3].set_clim((0, N_resamples * 2 / 3))
        ax.set_xticks(N_trials_one_polarity*2)
        ax.set_yticks(range(1,5))
        ax.set_xlabel('N trials')
        ax.set_ylabel('SD{threshold} (dB)')
        fig_handle = [fig_handle, fig_handle_lowN, f]
        for i,fh in enumerate(fig_handle):
            fh.savefig(figure_path.replace('.png',f'{i}.pdf'))
    else:
        xbins = np.hstack((N_trials_one_polarity,N_trials_one_polarity[-1]+np.diff(N_trials_one_polarity[:2])))-np.diff(N_trials_one_polarity[:2])/2
        ylims = (thresholds[:].min(),thresholds[:].max())
        if (ylims[0] >= (thresholds[-1,0]-7)) and (ylims[1] <= (thresholds[-1,0]+15)):
            ylims = [thresholds[-1,0]-7, thresholds[-1,0]+15]
            adj_lims = False
        else:
            adj_lims = True
            if ylims[0] == ylims[1]:
                ylims = 10*np.array([-1,1])+ylims
            else:
                ylims = np.diff(ylims)*.1*np.array([-1,1])+ylims
        ybins = np.linspace(ylims[0],ylims[1],20)
        fig_handle.axes[0].set_position([.07,.07,.4007,.63])
        fig_handle.axes[1].set_position([.5, .07, .5, .63])
        #plt.figure()
        ax = fig_handle.add_axes([.07,.75,.29,.25])
        ax2 = fig_handle.add_axes([.36, .75, .29, .25])
        ax3 = fig_handle.add_axes([.71, .75, .29, .25])
        h=ax.hist2d(np.tile(N_trials_one_polarity, (N_resamples, 1)).T.flatten(), thresholds.flatten(), bins=(xbins, ybins))
        # plt.colorbar(h[3])
        h[3].set_clim((0,N_resamples*2/5))
        ax.set_xlabel('Ntrials per subaverage')
        ax.set_ylabel('Threshold (dB SPL)')

        epochs_random = epochs.copy()
        signs = np.hstack((np.ones(256), -1 * np.ones(256)))
        noise_estimates = []
        for n in range(5):
            for level in levels:
                np.random.shuffle(signs)
                epochs_random.loc[(slice(None), level), :] = (epochs_random.loc[(slice(None),level),:] *
                                                        np.tile(signs, (epochs.shape[1], 1)).T)
            noise_estimates.append(np.sqrt(np.power(epochs_random.groupby('level').agg('mean').values,2).mean()))
        fit_results['noise_random_inv_RMS'] = np.mean(noise_estimates)
        ax.text(1, 1, f"Noise {fit_results['noise_random_inv_RMS']*1e6:.3f} uV", horizontalalignment='right',
             verticalalignment='top', transform=ax.transAxes, color='orange')
        if adj_lims:
            ax.text(1, .8, f"Adj lims", horizontalalignment='right',
                    verticalalignment='top', transform=ax.transAxes, color='orange')
        # f,ax2 = plt.subplots(1,3)
        # h=ax2.hist2d(np.tile(N_trials_one_polarity, (N_resamples, 1)).T.flatten(), thresholds.flatten(), bins=(xbins, ybins))
        # h[3].set_clim((0,N_resamples*2/5))
        h=ax2.hist2d(np.tile(N_trials_one_polarity, (N_resamples, 1)).T.flatten(), thresholds_by_resample_mean.flatten(), bins=(xbins, ybins))
        h[3].set_clim((0,N_resamples*2/5))
        ax2.set_yticklabels('')

        ylims2 = (thresholds_by_resample_std.min(), thresholds_by_resample_std.max())
        if (ylims2[0] >= 0) and (ylims2[1] <= 5):
            ylims2 = [0, 5]
            adj_lims = False
        else:
            adj_lims = True
            if ylims2[0] == ylims2[1]:
                ylims2 = 10 * np.array([-1, 1]) + ylims2
            else:
                ylims2 = np.diff(ylims2) * .1 * np.array([-1, 1]) + ylims2
        ybins2 = np.linspace(ylims2[0], ylims2[1], 25)

        h=ax3.hist2d(np.tile(N_trials_one_polarity, (N_resamples, 1)).T.flatten(), thresholds_by_resample_std.flatten(), bins=(xbins, ybins2))
        h[3].set_clim((0,N_resamples*2/5))
        if adj_lims:
            ax3.text(1, 1, f"Adj lims", horizontalalignment='right',
                    verticalalignment='top', transform=ax3.transAxes, color='orange')
        ax3.set_ylabel('std(threshold)')

    return fit_results, fig_handle


def estimate_threshold(epochs, seed=0, pst_range=None,
                       N_shuffles=500, avmode='median',
                       XC0m_threshold=0.3, save_data_resamples=False,
                       XC0m_plbounds=None, XC0m_sigbounds=None,
                       second_filter='pre-average',
                       second_filter_settings={'highpass': 300, 'lowpass': 3000, 'order': 1},
                       calc_XC0m_only=True,
                       KSs_plbounds=None, KSs_sigbounds=None,
                       XCp_near_0_plbounds=None, XCp_near_0_sigbounds=None, peak_lag_threshold=.5,
                       round_results=True, human_threshold=None,
                       plot_results=True,
                       keep_ind_masks=None,
                       fit_each_resample=False,
                       keep_fit_each_resample_fit_params=False):
    """
    This is code to algorithmically threshold ABR data as described in Shaheen et al 2024.
    Thresholds are generated by:
    1. Randomly splitting the trials into two groups, and calculating the median waveform for each group, followed by the normalized cross correlation between these median waveforms.
    2. This process is repeated 500 times to obtain a reshuffled cross-correlation distribution.
    3. The mean values of these distributions are computed for each level and fit with a sigmoid and a power law. The fit that provides the best mean squared error is then used, and threshold is defined as where that fit crossed a criterion (default 0.3).


Parameters
    ----------
    epochs : pd.DataFrame
        The index should contain (['polarity', 'level']).  Extra indexes will be dropped.
        Time must be in columns
    seed : int
        Seed used to initialize RandomState. Pass an integer for repeatable results.
    pst_range : list
        The pos-stimulus time range over which the cross-correlation is measured
    N_shuffles : int, default 500
        The number of times to repeat the random shuffling
    avmode : string, default 'median'
        If 'mean' create subaverages by mean
        If 'median' create them by median
    XC0m_threshold: float, default 0.3
        The criterion value of normalized cross correlation used to find threshold
    save_data_resamples: bool, default False
        If True, the results of each reshuffling are saved
    XC0m_plbounds: string, default None
        If 'increasing' sets bounds to make slope positive
    XC0m_sigbounds: string, default None
        If 'increasing, midpoint within one step of x range' sets bounds to make slope positive,
            # and midpoint within [min(level) - step, max(level) + step]. step is usually 5 dB
    second_filter: string, default None
        Applies an iir filter (forward and backward using scipy.signal.filtfilt) to each trial.
        If 'pre-avergae' applies the filter before averaging
        If 'post-avergae' applies the filter after averaging
        If None does not apply a filter.
    second_filter_settings: dict, default {'highpass': 300, 'lowpass': 3000, 'order': 1}
        The filter settings used for the second filter
    round_results: bool, default True
        If True, round results to 3 decimal places
    human_threshold : float
        The threshold as selected by a human rater. Not used in the algorithm, if passed will be indicated on the plots
    plot_results: bool, default True
        If True, plot results
    keep_ind_masks: tuple or list of arrays of bools, or None, default None
        If None, does nothing
        If list, should be length 2 (one for each polarity)
        Each array should be length of the number of trials in each level. False means to not use that trials
        Used to exclude a consistent set of trials to test effect of reducing SNR on threshold estimate
    fit_each_resample: bool, default False,
        If True, fit threshold for each resample set (all levels), returns extra fields in fit results:
            fit_results['thresholds'] array of thresholds, one for each resample
            fit_results['threshold_mean'] = mean of this array
            fit_results['threshold_std'] = std of this array
            fit_results['thresholds_nans'] = list of inidicies where fit failed
    keep_fit_each_resample_fit_params: bool, default False
            If this and fit_each_resample are True, keeps parameters for each fit (to be used for plotting)

    --The following parameters are used to control alternate ways of using the same cross-correlation distributions to
      find threshold. Empirically they didn't work as well.
    calc_XC0m_only : bool, default True
        Set to False to additionally calculcate (and plot) these two alternate ways.
    KSs_plbounds, KSs_sigbounds: Like above but used when using Kolmogorov-Smirnov test statistic to measure differences
        in cross correlation distribution between each level
    XCp_near_0_plbounds, XCp_near_0_sigbounds: Like above but used when using the percent of time peak cross correlation
        is near 0 for each level.
    peak_lag_threshold : float, default 0.5
        the time threshold (in ms) over which to calculate the percent of time peak cross correlation is near 0 (within +/-) this value

 Outputs
    ----------
    fit_results : dictionary of fitting results
    fig_handle : figure handle with plots of the results

    fit_results contains these keys:
        threshold: the threshold estimated by the algorithm.
            Set to -inf if all datapars['xc0mean'] are above XC0m_threshold*1.1 (multiplied by 1.1 to allow a small
              amount of extrapolation.
            Set to +inf if all datapars['xc0mean'] are below XC0m_threshold
            Set to np.nan if fitting find threshold fails (needs manual analysis)
        status:
            Success: fitting worked
            Failure: fitting failed
            Skipped: fitting skipped due to less than 100 trials for one of the stimuli
        status_message:
            If status was Success, will be one of:
              sigmoid: threshold was found with the sigmoid
              power low: threshold was found with the power law
              power law (noisy): threshold was found with the power law but was poorly fit. Review for accuracy.
            Otherwise will give details on why status is not Success
        fit_XC0m: details on the fit functions:
            sigmoid_fit: details on sigmoid fit
                params: fitting parameters
                yfit: reslt of fit
                sseL sum of squared errors
            power_law_fit: details on power law fit, parameters as for sigmoid
            bestFitType: Which function was chosed to find threshold
            algpars: criterion parameters used to find threshold
        algpars: parameters as passed in to estiamte_threshold
        datapars:
            fs: sampling frequency estimated from data
            levels: stimulus levels
            xc0mean: means of the cross correlation distributions (for 0 lag)
            xc0std:  standard deviations of the cross correlation distributions (for 0 lag)
            N_min_global: Minimum number of trials (per polarity) across all levels
        comptime: how long it took (in seconds) to run algorithm. First N elements are for each of N levels,
          last element is grand total
        plottime: how long it took (in seconds) to plot results.

       """
    t00 = time.time()
    N_subaverages = 2

    # Check that dataframe has the right indexes, drop unneeded indexes
    assert all([name in epochs.index.names for name in ['polarity', 'level']]), 'epochs dataframe must have ' \
                                                                        '"polarity" and level" as indexes'
    drop_these = [name for name in epochs.index.names if name not in ['polarity', 'level']]
    epochs.reset_index(drop_these, drop=True, inplace=True)
    epochs.index = epochs.index.reorder_levels(['polarity', 'level'])
    epochs.sort_index(inplace=True)

    # Drop empty trials
    all_values_0 = np.all(epochs == 0, axis=1)
    dropped_all = False
    if any(all_values_0):
        dropped_all = all(all_values_0)
        log.warning(f'\n {all_values_0.sum()}/{len(all_values_0)} epochs in this dataset have values of all 0.'
                    f' Dropping these epochs.')
        epochs = epochs[~all_values_0]

    #Count number of levels, polarities, trials per rep. Calculate fs (sampling frequency) for data)
    levels = epochs.index.get_level_values('level').unique().values
    polarities = epochs.index.get_level_values('polarity').unique().values
    unV, unN = np.unique(np.diff(epochs.keys().values), return_counts=True)
    fs = 1 / unV[unN.argmax()]
    # filter data if requested
    if second_filter == 'pre-average':
        epochs[:] = utils.filter(epochs.values, fs, **second_filter_settings)

    peak_lag_threshold_samples = np.ceil(peak_lag_threshold * fs / 1000)
    N_by_polarity = epochs.groupby(['polarity', 'level']).size()
    if len(N_by_polarity) == 0:
        if dropped_all:
            N_min_global = 0
        else:
            raise RuntimeError('N_by_polarity is empty but not all epochs were dropped. If this happened its a bug.')
    else:
        N_min_global = N_by_polarity.min()
    if N_min_global < 100:
        if dropped_all:
            msg = f'All epochs were dropped because they all had values of all zero.'
        else:
            msg = f'There is at least one condition with only {N_min_global} reps. Minimum needed (arbitrary) is 100.'\
              f'There are {len(epochs)} total reps for this recording'
        log.warning(msg)
        datapars = {'fs': fs, 'levels': levels}
        algpars = {'peak_lag_threshold': peak_lag_threshold,
                   'N_shuffles': N_shuffles}
        fit_results = {'threshold': np.nan, 'threshold_XCp_near_0': np.nan,
                       'fit_XCp_near_0': None, 'fit_KSs': None,
                       'peak_lag': np.nan, 'peak': np.nan, 'xc0': np.nan,
                       'xcLm': np.nan, 'xcLstd': np.nan,
                       'algpars': algpars, 'datapars': datapars}
        fit_results['threshold_XC0m'] = np.nan
        fit_results['threshold_XCp_near_0'] = np.nan
        fit_results['threshold_KSs'] = np.nan
        fit_results['fit_XCp_near_0'] = None
        fit_results['fit_KSs'] = None
        fit_results['fit_XC0m'] = None
        fit_results['comptime'] = np.nan
        fit_results['status'] = 'Skipped'
        fit_results['status_message'] = msg
        return fit_results, plt.figure()

    N_varies = len(N_by_polarity.unstack('polarity').min(axis=1).unique()) > 1
    if N_varies:
        log.warning('\n Different level combinations have different number of reps. '
                    'The number of reps per subaverage will differ by level.')
    else:
        N_per_group = int(np.floor(N_min_global / N_subaverages))

    # Calculate time window to compute correlation over
    if pst_range is None:
        N_time = len(epochs.keys())
        time_inds = np.full(N_time, True)
    else:
        time_inds = (epochs.keys().values >= pst_range[0]) & (
                epochs.keys().values < pst_range[1])
        N_time = time_inds.sum()

    # Initialize variables
    if calc_XC0m_only:
        xc0 = np.zeros((len(levels), N_shuffles))
    else:
        xc = np.zeros((len(levels), N_shuffles, N_time))
        peak_index = np.zeros((len(levels), N_shuffles), dtype=int)
        peak_lag = np.zeros((len(levels), N_shuffles), dtype=int)
        peak = np.zeros((len(levels), N_shuffles))
        lags = np.arange(N_time) - (N_time - 1) / 2
        L0 = np.where(lags == 0)[0][0]
    epochs_means = np.zeros((len(epochs.keys()), len(levels), 3))
    epochs_sems = np.zeros((len(epochs.keys()), len(levels)))
    ABRtime = epochs.keys().values.astype('float') * 1000
    rn = np.random.RandomState(seed)
    comptime = np.zeros(len(levels)+1)

    # Cross-correlate subaverages and store in a nd array for each level
    for i, level in enumerate(levels):
        t0 = time.time()
        #initialize variables for this level
        if N_varies:
            N_min = N_by_polarity.loc[:, level].min()
            N_per_group = int(np.floor(N_min / N_subaverages))
        # epochs_i = epochs.loc[:, :, level, :]
        # indP = epochs.loc[:, 1, level, :].index.get_level_values(0).values
        # indN = epochs.loc[:, -1, level, :].index.get_level_values(0).values
        epochsP = epochs.loc[(1, level), time_inds]
        epochsN = epochs.loc[(-1, level), time_inds]
        indP = np.arange(len(epochsP))
        indN = np.arange(len(epochsN))
        if keep_ind_masks is not None:
            indP = indP[keep_ind_masks[0]]
            indN = indN[keep_ind_masks[1]]
            N_per_group = int(min((len(indP),len(indN))) / N_subaverages)
        randomized_indicesP = np.zeros((N_per_group * N_subaverages, N_shuffles), dtype=int)
        randomized_indicesN = np.zeros((N_per_group * N_subaverages, N_shuffles), dtype=int)

        # loop across number of shuffles
        for ishuf in range(N_shuffles):
            # randomize into two buckets, evenly splitting by polarity
            randomized_indicesP[:, ishuf] = rn.choice(indP, N_per_group * N_subaverages, replace=False)
            randomized_indicesN[:, ishuf] = rn.choice(indN, N_per_group * N_subaverages, replace=False)

            # average (either mean or median)
            if avmode == 'mean':
                epochs_mean0 = epochsP.iloc[randomized_indicesP[:N_per_group, ishuf]].mean() + \
                               epochsN.iloc[randomized_indicesN[:N_per_group, ishuf]].mean()
                epochs_mean1 = epochsP.iloc[randomized_indicesP[N_per_group:, ishuf]].mean() + \
                               epochsN.iloc[randomized_indicesN[N_per_group:, ishuf]].mean()
            elif avmode == 'median':
                epochs_mean0 = np.median(np.vstack((epochsP.iloc[randomized_indicesP[:N_per_group, ishuf]].values,
                                                    epochsN.iloc[randomized_indicesN[:N_per_group, ishuf]].values)),
                                         axis=0)
                epochs_mean1 = np.median(np.vstack((epochsP.iloc[randomized_indicesP[N_per_group:, ishuf]].values,
                                                    epochsN.iloc[randomized_indicesN[N_per_group:, ishuf]].values)),
                                         axis=0)

            else:
                raise RuntimeError(f'Invalid avmode: {avmode}')
            # filter data if requested
            if second_filter == 'post-average':
                epochs_mean0 = utils.filter(epochs_mean0, fs, **second_filter_settings)
                epochs_mean1 = utils.filter(epochs_mean1, fs, **second_filter_settings)
            #calculate cross correlation (for 0 lag only if that's all that's needed)
            if calc_XC0m_only:
                xc0[i, ishuf] = np.corrcoef(epochs_mean0, epochs_mean1)[0, 1]
            else:
                xc[i, ishuf, :] = utils.crossCorr(epochs_mean0, epochs_mean1, norm=True)
        if not calc_XC0m_only:
            peak_index[i, :] = np.argmax(xc[i, :, :], axis=-1)
            peak_lag[i, :] = peak_index[i, :] - (N_time - 1) / 2
            peak[i, :] = np.max(xc[i, :, :], axis=-1)

        #calculate mean and std of waveform for this level
        epochs_means[:, i, 0] = epochs.xs(level, level='level').values.mean(axis=0)
        epochs_sems[:, i] = np.std(epochs.xs(level, level='level').values, axis=0, ddof=1) / \
                            np.sqrt(len(indP) + len(indN))
        # find which shuffle yielded a cross-corraltion value closest to the mean
        if calc_XC0m_only:
            ishuf_mean = np.argmin(np.abs(xc0[i, :] - xc0[i, :].mean()))
        else:
            ishuf_mean = np.argmin(np.abs(xc[i, :, L0] - xc[i, :, L0].mean()))
        # Create and save subaverages for this shuffle to show on plot later
        if avmode == 'mean':
            epochs_means[:, i, 1] = np.mean(
                np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[:N_per_group, ishuf_mean]].values,
                            epochs.loc[(-1, level)].iloc[randomized_indicesN[:N_per_group, ishuf_mean]].values)),
                axis=0)
            epochs_means[:, i, 2] = np.mean(
                np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[N_per_group:, ishuf_mean]].values,
                           epochs.loc[(-1, level)].iloc[randomized_indicesN[N_per_group:, ishuf_mean]].values)),
                axis=0)
        elif avmode == 'median':
            epochs_means[:, i, 1] = np.median(
                np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[:N_per_group, ishuf_mean]].values,
                           epochs.loc[(-1, level)].iloc[randomized_indicesN[:N_per_group, ishuf_mean]].values)),
                axis=0)
            epochs_means[:, i, 2] = np.median(
                np.vstack((epochs.loc[(1, level)].iloc[randomized_indicesP[N_per_group:, ishuf_mean]].values,
                           epochs.loc[(-1, level)].iloc[randomized_indicesN[N_per_group:, ishuf_mean]].values)),
                axis=0)
        if second_filter == 'post-average':
            epochs_means[:, i, 0] = utils.filter(epochs_means[:, i, 0], fs, **second_filter_settings)
            epochs_means[:, i, 1] = utils.filter(epochs_means[:, i, 1], fs, **second_filter_settings)
            epochs_means[:, i, 2] = utils.filter(epochs_means[:, i, 2], fs, **second_filter_settings)
        t1 = time.time()
        comptime[i] = t1 - t0
    if not calc_XC0m_only:
        p_near_0 = (np.abs(peak_lag[:i + 1, :]) < peak_lag_threshold_samples).sum(axis=1) / N_shuffles
        xc0 = xc[:, :, L0]

    xc0mean = xc0.mean(axis=1)
    xc0std = xc0.std(axis=1)
    # fit with a sigmoid and power law, also contains logic to decide which fit to use
    fit_XC0m, threshold_XC0m = utils.fit_sigmoid_power_law(levels, xc0mean, XC0m_threshold, y_err=xc0std,
                                                           sigbounds=XC0m_sigbounds, plbounds=XC0m_plbounds)

    if fit_each_resample:
        thresholds = np.zeros(N_shuffles)
        thresholds_nans = []
        if keep_fit_each_resample_fit_params:
            sigmoid_fit_params = []
        else:
            sigmoid_fit_params = None
        for ishuf in range(N_shuffles):
            if fit_XC0m['bestFitType'] == 'sigmoid':
                sigmoid_fit_params_ = utils.fit_sigmoid(levels, xc0[:,ishuf], bounds=XC0m_sigbounds, calc_yfit=False)
                thresholds[ishuf] = utils.sigmoid_get_threshold(XC0m_threshold, *sigmoid_fit_params_)
                if keep_fit_each_resample_fit_params:
                    sigmoid_fit_params.append(sigmoid_fit_params_)
                if np.isnan(thresholds[ishuf]):
                    warnings.warn(f'Threshold is nan for shuffle {ishuf}/{N_shuffles}, running full fitter')
                    fit_XC0m_, threshold_XC0m_ = utils.fit_sigmoid_power_law(levels, xc0[:, ishuf], XC0m_threshold,
                                                                             sigbounds=XC0m_sigbounds,
                                                                             plbounds=XC0m_plbounds)
                    thresholds[ishuf] = threshold_XC0m_
                    thresholds_nans.append(ishuf)
            elif fit_XC0m['bestFitType'] in ['power law', 'power law (noisy)']:
                power_law_fit_params = utils.fit_power_law(levels, xc0[:, ishuf], bounds=XC0m_plbounds,
                                                       calc_yfit=False)
                thresholds[ishuf] = utils.power_law_get_threshold(XC0m_threshold, *power_law_fit_params)
            elif fit_XC0m['bestFitType'] in ['all below criterion, threshold is inf',
                                             "most below criterion, but couldn't fit, threshold is inf",
                                             'all above criterion, threshold is -inf', None]:
                fit_XC0m_, threshold_XC0m_ = utils.fit_sigmoid_power_law(levels, xc0[:, ishuf], XC0m_threshold,
                                                                         sigbounds=XC0m_sigbounds,
                                                                         plbounds=XC0m_plbounds)
                thresholds[ishuf] = threshold_XC0m_
            else:
                import pdb
                pdb.set_trace()
                raise RuntimeError(f"fix. bestFitType is {fit_XC0m['bestFitType']}")
            a=2

        thresholds.mean()
    # If also finding threshold by alternate methods, fit those here
    if not calc_XC0m_only:
        nulld = xc[0, :, L0]
        XCpeak_pv = np.zeros(len(levels))
        XCpeak_KSs = np.zeros(len(levels))
        for i, level in enumerate(levels):
            res = scipy.stats.kstest(nulld, xc0[i, :], alternative='greater')
            XCpeak_pv[i] = res.pvalue
            XCpeak_KSs[i] = res.statistic

        thresholdCriterion = .3
        fit_KSs, threshold_KSs = utils.fit_sigmoid_power_law(levels, XCpeak_KSs, thresholdCriterion,
                                                             sigbounds=KSs_sigbounds, plbounds=KSs_plbounds)

        thresholdCriterion2 = .4
        fit_XCp_near_0, threshold_XCp_near_0 = utils.fit_sigmoid_power_law(levels, p_near_0, thresholdCriterion2,
                                                                           sigbounds=XCp_near_0_sigbounds,
                                                                           plbounds=XCp_near_0_plbounds)

    # Find levels_, a selection of levels used to plot (showing them all makes the figure too hard to read)
    threshold_level = threshold_XC0m
    if np.isinf(threshold_level):
        if threshold_level > 0:
            levels_ = levels[-12:]
        else:
            levels_ = levels[:12]
    elif np.isnan(threshold_level):
        levels_ = levels[-12:]
    else:
        try:
            threshold_level = levels[np.where(threshold_level > levels)[0][-1]]
        except:
            threshold_level = levels[np.min((len(levels)-1, 5))]
        levels_ = levels[(levels >= (threshold_level - 40)) & (levels <= (threshold_level + 30))]

    # Store results in dictionaries
    datapars = {'fs': fs, 'levels': levels, 'xc0mean': xc0mean, 'xc0std': xc0std, 'N_min_global': N_min_global}
    if save_data_resamples:
        # Only save these if requested
        datapars.update({'xc0': xc0})
        if not calc_XC0m_only:
            datapars.update({'peak_lag': peak_lag, 'peak': peak, 'xcLmean': xc.mean(axis=1),
                             'xcLstd': xc.std(axis=1)})

    if N_varies:
        datapars.update({'N_by_polarity': N_by_polarity.unstack('polarity').values.T})

    algpars = {'peak_lag_threshold': peak_lag_threshold,
               'N_shuffles': N_shuffles}
    fit_results = {'threshold': threshold_XC0m}
    if fit_XC0m['thdEstimationFailed']:
        fit_results['status'] = 'Failure'
        fit_results['status_message'] = 'curve fitting did not cross threshold'
    else:
        fit_results['status'] = 'Success'
        fit_results['status_message'] = fit_XC0m['bestFitType']

    fit_results['fit_XC0m'] = fit_XC0m

    if fit_each_resample:
        fit_results['threshold_mean'] = np.round(thresholds.mean(), 2)
        fit_results['threshold_std'] = np.round(thresholds.std(), 2)
        fit_results['thresholds'] = np.round(thresholds, 2)
        fit_results['thresholds_nans'] = thresholds_nans
        if keep_fit_each_resample_fit_params:
            fit_results['sigmoid_fit_params'] = sigmoid_fit_params

    if not calc_XC0m_only:
        fit_results['threshold_XCp_near_0'] = threshold_XCp_near_0
        fit_results['threshold_KSs'] = threshold_KSs
        fit_results['fit_XCp_near_0'] = fit_XCp_near_0
        fit_results['fit_KSs'] = fit_KSs

    fit_results['algpars'] = algpars
    fit_results['datapars'] = datapars
    t1 = time.time()
    comptime[-1] = t1 - t00
    fit_results['comptime'] = comptime

    if any(all_values_0):
        fit_results['status_message'] += f' WARNING: {all_values_0.sum()}/{len(all_values_0)} epochs had values of ' \
                                         f'all 0 and were dropped.'
    # Plot results
    t0 = time.time()
    if plot_results:
        if calc_XC0m_only:
            if fit_each_resample:
                fig_handle = plot_fit_each_resample(levels, levels_, xc0, ABRtime, epochs_means,
                       epochs_sems, pst_range, fit_XC0m, thresholds=thresholds, norm_waveforms=True,
                       human_threshold=human_threshold, avmode=avmode, criterion=XC0m_threshold, fit_params=sigmoid_fit_params)
            else:
                fig_handle = plot_fit(levels, levels_, xc0, ABRtime, epochs_means,
                       epochs_sems, pst_range, fit_XC0m, norm_waveforms=True,
                       human_threshold=human_threshold, avmode=avmode, criterion=XC0m_threshold)
        else:
            fig_handle = plot_fit_full(lags, levels, levels_, xc, xc0, peak_lag, ABRtime, epochs_means,
                                epochs_sems, pst_range, p_near_0,  XCpeak_pv,
                                XCpeak_KSs, fit_KSs, fit_XCp_near_0, fit_XC0m, norm_waveforms=True,
                                human_threshold=human_threshold, avmode=avmode)
        fit_results['plottime'] = time.time() - t0
    else:
        fig_handle = None

    if round_results:
        try:
            fit_results['threshold'] = np.round(fit_results['threshold'], 2)
            fit_results['fit_XC0m']['threshold']= np.round(fit_results['fit_XC0m']['threshold'],2)
            fit_results['fit_XC0m']['sigmoid_fit']['yfit']= np.round(fit_results['fit_XC0m']['sigmoid_fit']['yfit'],3)
            fit_results['fit_XC0m']['sigmoid_fit']['sse'] = np.round(fit_results['fit_XC0m']['sigmoid_fit']['sse'],4)
            fit_results['fit_XC0m']['power_law_fit']['yfit'] = np.round(fit_results['fit_XC0m']['power_law_fit']['yfit'], 3)
            fit_results['fit_XC0m']['power_law_fit']['sse'] = np.round(fit_results['fit_XC0m']['power_law_fit']['sse'], 4)
            fit_results['fit_XC0m']['power_law_fit']['adj_r2'] = np.round(fit_results['fit_XC0m']['power_law_fit']['adj_r2'], 4)
            fit_results['datapars']['xc0mean'] = np.round(fit_results['datapars']['xc0mean'],3)
            fit_results['datapars']['xc0std'] = np.round(fit_results['datapars']['xc0std'], 3)
            fit_results['comptime'] = np.round(fit_results['comptime'], 3)
            fit_results['plottime'] = np.round(fit_results['plottime'], 3)
        except:
            pass
    return fit_results, fig_handle


def plot_fit(levels, levels_, xc0, ABRtime, epochs_means, epochs_sems, pst_range, fit_XC0m,
             norm_waveforms=True, human_threshold=None, avmode='mean', criterion=0.3):
    # In the left column the figures show mean +/- SE of all trials in black, and median (or mean, depending on AVmode)
    # for the two subsets. Waveforms are normalized (for each level all 3 lines are scaled by the peak-to-peak of
    # the mean of all trials). The right hand side shows mean correlation coefficient vs stimulus level. Sigmoid and
    # power law fits to this data are shown in green and purple. The threshold is shown by the pink dashed line.

    fs_scale = 1
    fs_labels = 10*fs_scale
    fs_ticklabels = 10*fs_scale

    udiffs, counts = np.unique(np.diff(np.array(levels)), return_counts=True)
    m = counts.argmax()
    level_diff_mode = udiffs[m]

    if pst_range is not None:
        ii = (ABRtime > pst_range[0]*1000) & (ABRtime <= pst_range[1]*1000)
    else:
        ii = np.full(ABRtime.shape, True)

    fig_handle, ax = plt.subplots(1, 2, figsize=(7, 10), gridspec_kw={'hspace': 0.07, 'wspace': .25, 'top': 1, 'bottom': 0.07,
                                                               'left': .07, 'right': 1, 'width_ratios': [.4, .6]})
    for level in levels_:
        i = np.where(level == levels)[0][0]
        if epochs_means is not None:
            y = epochs_means[:, i, 0].T * 1e6  # normalizing to full waveform

            if norm_waveforms:
                ys = 1 / (y[ii].max(axis=0) - y[ii].min(axis=0))
            else:
                ys = 1
            # plotting just windowed data
            h0 = ax[0].plot(ABRtime[ii], y[ii] * ys * level_diff_mode + level, 'k', linewidth=1)

            h0f = ax[0].fill_between(ABRtime[ii],
                        np.squeeze(epochs_means[ii, i, 0] - epochs_sems[ii, i]) * 1e6 * ys * level_diff_mode + level,
                        np.squeeze(epochs_means[ii, i, 0] + epochs_sems[ii, i]) * 1e6 * ys * level_diff_mode + level,
                        color='lightgrey', alpha=.5)

            h1 = ax[0].plot(ABRtime[ii], epochs_means[ii, i, 1] * 1e6 * ys * level_diff_mode + level, color=colors[0],
                       linewidth=1)
            h2 = ax[0].plot(ABRtime[ii], epochs_means[ii, i, 2] * 1e6 * ys * level_diff_mode + level, color=colors[1],
                       linewidth=1)
            if pst_range is not None:
                ax[0].set_xlim(np.array(pst_range) * 1000)
    if norm_waveforms:
        ax[0].set_ylim((levels_[0]-level_diff_mode*.7, levels_[-1]+ 0.7*level_diff_mode + 0.09*(levels_[-1]-levels_[0])))
    ax[0].set_xlabel('Time (ms)', fontsize=fs_labels)
    ax[0].set_ylabel('Level (dB SPL)', fontsize=fs_labels)
    ax[0].legend([(h0[0], h0f), h1[0], h2[0]], ['mean \u00B1 SE of all trials', avmode +' of 1st subset',
                                         avmode +' of 2nd subset'], loc='upper left', bbox_to_anchor=(0,1))
    ax[0].set_yticks(levels_)
    time_lines = np.array([2, 4, 6, 8])
    if pst_range is not None:
        time_lines = time_lines[(time_lines > pst_range[0]*1000) & (time_lines < pst_range[1]*1000)]
    for tl in time_lines:
        ax[0].axvline(tl, color='lightgrey', zorder=-10, linewidth=.5)

    # ax[0].set_ylabel('Level (dB SPL)')

    if fit_XC0m is not None:
        # levi = [i for i,lev in enumerate(levels) if any(lev==np.array([30,65]))]
        # h = ax[1].violinplot(xc0[levi,:].T, levels[levi], widths=np.diff(levels[:2])[0]*.9, showextrema=False, points=200,
        #                    showmeans=False)
        h = ax[1].violinplot(xc0.T, levels, widths=np.diff(levels[:2])[0]*.9, showextrema=False, points=200,
                           showmeans=False)
        for pc in h['bodies']:
            pc.set_facecolor('lightgrey')
            # pc.set_edgecolor('black')
            pc.set_alpha(.5)
        ax[1].plot(levels, xc0.mean(axis=1), '.k', label='data mean', )
        # ax[1].plot(levels[levi], xc0[levi,:].mean(axis=1), '.k', label='data mean', )
        ax[1].set_ylim((-.3, 1.01))
        yf = None
        if fit_XC0m['sigmoid_fit'] is not None:
            if fit_XC0m['bestFitType'] == 'sigmoid':
                l = f"sigmoid fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                yf = fit_XC0m['sigmoid_fit']['yfit']
            else:
                l = f"sigmoid fit,\nnot used"
                lw = 1

            ax[1].plot(levels, fit_XC0m['sigmoid_fit']['yfit'], color=colors[2], lw=lw, label=l)
            if fit_XC0m['bestFitType'] == 'power law':
                l = f"power law fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                ls = '-'
                yf = fit_XC0m['power_law_fit']['yfit']
            elif fit_XC0m['bestFitType'] == 'power law (noisy)':
                l = f"Noisy power law fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                ls = '--'
                yf = fit_XC0m['power_law_fit']['yfit']
            else:
                l = f"power law fit,\nnot used"
                lw = 1
                ls = '-'
            ax[1].plot(levels, fit_XC0m['power_law_fit']['yfit'], color=colors[4], lw=lw, label=l, ls=ls)

        if fit_XC0m['threshold'] is not None:
            if fit_XC0m['threshold'] is np.inf:
                l = 'thresh=inf\n(all levels below criterion)'
            elif np.isnan(fit_XC0m['threshold']):
                l = 'thresh is nan\n(error fitting?)'
            else:
                l = None
            ax[1].axvline(fit_XC0m['threshold'], color=colors[6], linestyle='--', linewidth=1, label=l)
            time_range = np.array((ABRtime[ii][0], ABRtime[ii][-1]))
            ax[0].plot((time_range[0]-np.diff(time_range)*.25, time_range[1]+np.diff(time_range)*.05),
                       fit_XC0m['threshold']*np.ones(2), color=colors[6], linestyle='--', linewidth=1)
            ax[0].set_clip_on(False)
        ax[1].axhline(criterion, color='k', linestyle='--', linewidth=.5)
        ax[1].text(ax[1].get_xlim()[1], criterion, 'criterion', horizontalalignment='right', fontsize=10*fs_scale)
        ax[1].set_ylabel('Correlation Coefficient', fontsize=fs_labels)
        ax[1].set_yticks((0, .2, .4, .6, .8, 1))
        ax[1].set_yticklabels(('0', '.2', '.4', '.6', '.8', '1'))
        ax[1].set_xlabel('Level (dB SPL)', fontsize=fs_labels)
        levi = (np.abs(levels-np.percentile(ax[1].get_xlim(),35))).argmin()
        if (yf is None) or yf[levi] < .8:
            ax[1].legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fontsize=10 * fs_scale, framealpha=1)
        else:
            ax[1].legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=True, fontsize=10 * fs_scale, framealpha=1)
        ax[0].tick_params(labelsize=fs_ticklabels)
        ax[1].tick_params(labelsize=fs_ticklabels)
        ax[1].set_clip_on(False)
    # Hide the right and top spines
    [ax_.spines.right.set_visible(False) for ax_ in ax]
    [ax_.spines.top.set_visible(False) for ax_ in ax]

    if human_threshold is not None:
        ax[1].axvline(human_threshold, color='k', linestyle='--', linewidth=1)
    return fig_handle

def plot_fit_each_resample(levels, levels_, xc0, ABRtime, epochs_means, epochs_sems, pst_range, fit_XC0m,
             thresholds=None, norm_waveforms=True, human_threshold=None, avmode='mean', criterion=0.3, fit_params=None):
    # In the left column the figures show mean +/- SE of all trials in black, and median (or mean, depending on AVmode)
    # for the two subsets. Waveforms are normalized (for each level all 3 lines are scaled by the peak-to-peak of
    # the mean of all trials). The right hand side shows mean correlation coefficient vs stimulus level. Sigmoid and
    # power law fits to this data are shown in green and purple. The threshold is shown by the pink dashed line.

    fs_scale = 1
    fs_labels = 10*fs_scale
    fs_ticklabels = 10*fs_scale

    udiffs, counts = np.unique(np.diff(np.array(levels)), return_counts=True)
    m = counts.argmax()
    level_diff_mode = udiffs[m]

    if pst_range is not None:
        ii = (ABRtime > pst_range[0]*1000) & (ABRtime <= pst_range[1]*1000)
    else:
        ii = np.full(ABRtime.shape, True)

    fig_handle, ax = plt.subplots(1, 2, figsize=(7, 10), gridspec_kw={'hspace': 0.07, 'wspace': .25, 'top': 1, 'bottom': 0.07,
                                                               'left': .07, 'right': 1, 'width_ratios': [.4, .6]})
    for level in levels_:
        i = np.where(level == levels)[0][0]
        if epochs_means is not None:
            y = epochs_means[:, i, 0].T * 1e6  # normalizing to full waveform

            if norm_waveforms:
                ys = 1 / (y[ii].max(axis=0) - y[ii].min(axis=0))
            else:
                ys = 1
            # plotting just windowed data
            h0 = ax[0].plot(ABRtime[ii], y[ii] * ys * level_diff_mode + level, 'k', linewidth=1)

            h0f = ax[0].fill_between(ABRtime[ii],
                        np.squeeze(epochs_means[ii, i, 0] - epochs_sems[ii, i]) * 1e6 * ys * level_diff_mode + level,
                        np.squeeze(epochs_means[ii, i, 0] + epochs_sems[ii, i]) * 1e6 * ys * level_diff_mode + level,
                        color='lightgrey', alpha=.5)

            h1 = ax[0].plot(ABRtime[ii], epochs_means[ii, i, 1] * 1e6 * ys * level_diff_mode + level, color=colors[0],
                       linewidth=1)
            h2 = ax[0].plot(ABRtime[ii], epochs_means[ii, i, 2] * 1e6 * ys * level_diff_mode + level, color=colors[1],
                       linewidth=1)
            if pst_range is not None:
                ax[0].set_xlim(np.array(pst_range) * 1000)
    if norm_waveforms:
        ax[0].set_ylim((levels_[0]-level_diff_mode*.7, levels_[-1]+ 0.7*level_diff_mode + 0.09*(levels_[-1]-levels_[0])))
    ax[0].set_xlabel('Time (ms)', fontsize=fs_labels)
    ax[0].set_ylabel('Level (dB SPL)', fontsize=fs_labels)
    ax[0].legend([(h0[0], h0f), h1[0], h2[0]], ['mean \u00B1 SE of all trials', avmode +' of 1st subset',
                                         avmode +' of 2nd subset'], loc='upper left', bbox_to_anchor=(0,1))
    ax[0].set_yticks(levels_)
    time_lines = np.array([2, 4, 6, 8])
    if pst_range is not None:
        time_lines = time_lines[(time_lines > pst_range[0]*1000) & (time_lines < pst_range[1]*1000)]
    for tl in time_lines:
        ax[0].axvline(tl, color='lightgrey', zorder=-10, linewidth=.5)

    # ax[0].set_ylabel('Level (dB SPL)')

    if fit_XC0m is not None:
        # levi = [i for i,lev in enumerate(levels) if any(lev==np.array([30,65]))]
        # h = ax[1].violinplot(xc0[levi,:].T, levels[levi], widths=np.diff(levels[:2])[0]*.9, showextrema=False, points=200,
        #                    showmeans=False)
        h = ax[1].violinplot(xc0.T, levels, widths=np.diff(levels[:2])[0]*.9, showextrema=False, points=200,
                           showmeans=False)
        for pc in h['bodies']:
            pc.set_facecolor('lightgrey')
            # pc.set_edgecolor('black')
            pc.set_alpha(.5)
        ax[1].plot(levels, xc0.mean(axis=1), '.k', label='data mean', )
        # ax[1].plot(levels[levi], xc0[levi,:].mean(axis=1), '.k', label='data mean', )
        ax[1].set_ylim((-.3, 1.01))
        yf = None
        if fit_XC0m['sigmoid_fit'] is not None:
            if fit_XC0m['bestFitType'] == 'sigmoid':
                l = f"sigmoid fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                yf = fit_XC0m['sigmoid_fit']['yfit']
            else:
                l = f"sigmoid fit,\nnot used"
                lw = 1

            ax[1].plot(levels, fit_XC0m['sigmoid_fit']['yfit'], color=colors[2], lw=lw, label=l)
            if fit_XC0m['bestFitType'] == 'power law':
                l = f"power law fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                ls = '-'
                yf = fit_XC0m['power_law_fit']['yfit']
            elif fit_XC0m['bestFitType'] == 'power law (noisy)':
                l = f"Noisy power law fit,\nthresh={fit_XC0m['threshold']:.1f}"
                lw = 2
                ls = '--'
                yf = fit_XC0m['power_law_fit']['yfit']
            else:
                l = f"power law fit,\nnot used"
                lw = 1
                ls = '-'
            ax[1].plot(levels, fit_XC0m['power_law_fit']['yfit'], color=colors[4], lw=lw, label=l, ls=ls)

        if fit_params is not None:
            levs = np.arange(levels[0], levels[-1] + .1, .1)
            for fit_param in fit_params:
                yfit = utils.sigmoid(levs, *fit_param)
                ax[1].plot(levs, yfit, color=colors[2]+np.array([0,.15,0]), lw=.2, ls='-', alpha=.1)

        if fit_XC0m['threshold'] is not None:
            if fit_XC0m['threshold'] is np.inf:
                l = 'thresh=inf\n(all levels below criterion)'
            elif np.isnan(fit_XC0m['threshold']):
                l = 'thresh is nan\n(error fitting?)'
            else:
                l = None
            ax[1].axvline(fit_XC0m['threshold'], color=colors[6], linestyle='--', linewidth=1, label=l)
            time_range = np.array((ABRtime[ii][0], ABRtime[ii][-1]))
            ax[0].plot((time_range[0]-np.diff(time_range)*.25, time_range[1]+np.diff(time_range)*.05),
                       fit_XC0m['threshold']*np.ones(2), color=colors[6], linestyle='--', linewidth=1)
            ax[0].set_clip_on(False)
            if thresholds is not None:
                h = ax[1].violinplot(thresholds, [criterion], widths=[.1],showextrema=False, points=200,
                                     showmeans=False, vert=False)# mpl 3.10 on use: orientation='horizontal')
                for pc in h['bodies']:
                    pc.set_facecolor('red')
                    # pc.set_edgecolor('black')
                    pc.set_alpha(.5)

        ax[1].axhline(criterion, color='k', linestyle='--', linewidth=.5)
        ax[1].text(ax[1].get_xlim()[1], criterion, 'criterion', horizontalalignment='right', fontsize=10*fs_scale)
        ax[1].set_ylabel('Correlation Coefficient', fontsize=fs_labels)
        ax[1].set_yticks((0, .2, .4, .6, .8, 1))
        ax[1].set_yticklabels(('0', '.2', '.4', '.6', '.8', '1'))
        ax[1].set_xlabel('Level (dB SPL)', fontsize=fs_labels)
        levi = (np.abs(levels-np.percentile(ax[1].get_xlim(),35))).argmin()
        if (yf is None) or yf[levi] < .8:
            ax[1].legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fontsize=10 * fs_scale, framealpha=1)
        else:
            ax[1].legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=True, fontsize=10 * fs_scale, framealpha=1)
        ax[0].tick_params(labelsize=fs_ticklabels)
        ax[1].tick_params(labelsize=fs_ticklabels)
        ax[1].set_clip_on(False)
    # Hide the right and top spines
    [ax_.spines.right.set_visible(False) for ax_ in ax]
    [ax_.spines.top.set_visible(False) for ax_ in ax]

    if human_threshold is not None:
        ax[1].axvline(human_threshold, color='k', linestyle='--', linewidth=1)
    return fig_handle

def plot_fit_full(lags, levels, levels_, xc, xc0, peak_lag, ABRtime, epochs_means, epochs_sems, pst_range, p_near_0,
                  XCpeak_pv, XCpeak_KSs, fit_KSs, fit_XCp_near_0, fit_XC0m=None, norm_waveforms=True,
                  human_threshold=None, avmode='mean', presentation=False):
    # Full plot of results. Also plots two other ways to measure threshold from the correlation distributions that
    # empirically didn't work as well. Not as polished as plot_fit

    fig_handle = plt.figure(constrained_layout=True, figsize=(22, 14)) # np.array((22,10))/1.6) # (22,14)
    subfigs = fig_handle.subfigures(1, 2, wspace=0.07, width_ratios=[3, 1.]) #wspace=0.07
    ax = subfigs[0].subplots(len(levels_), 4, sharex='col', sharey='col', gridspec_kw={'hspace':0, 'top':0.99, 'bottom':0.03})
    ax = np.flipud(ax)
    for axi,level in enumerate(levels_):
        i = np.where(level == levels)[0][0]
        if epochs_means is not None:
            y = epochs_means[:, i, 0].T * 1e6
            if norm_waveforms:
                ys = 1/np.max(np.abs(y))
            else:
                ys = 1
            ax[axi, 0].plot(ABRtime, y*ys, 'k')

            ax[axi, 0].fill_between(ABRtime,
                                  np.squeeze(epochs_means[:, i, 0] - epochs_sems[:, i]) * 1e6*ys,
                                  np.squeeze(epochs_means[:, i, 0] + epochs_sems[:, i]) * 1e6*ys, color='grey')
            ax[axi, 0].plot(ABRtime, epochs_means[:, i, 1] * 1e6*ys, color='tab:blue')
            ax[axi, 0].plot(ABRtime, epochs_means[:, i, 2] * 1e6*ys, color='tab:orange')
            # if pst_range is not None:
            #     ax[axi,0].set_xlim(np.array(pst_range)*1000)

        if xc is not None:
            mean = xc[i, :, :].mean(axis=0)
            std = xc[i, :, :].std(axis=0)
            ax[axi, 1].fill_between(lags, mean - std, mean + std, color='k', alpha=.5, zorder=1e6)
            # ax[axi, 1].plot(lags, xc[i, :, :].T, linewidth=.5)
            ax[axi, 1].plot(lags, mean,'k',linewidth=3)

        ax[axi, 2].hist(peak_lag[i, :], np.linspace(lags[0], lags[-1], 40))

        ax[axi, 3].hist(xc0[i,:], np.linspace(-.5, 1, 60))
        # ax[axi, 3].set_ylabel(f'{level:.0f} dB', rotation='horizontal')
        # ax[axi, 3].yaxis.set_label_coords(1.1, .5)
        ax[axi, 0].set_ylabel(f'{level:.0f} dB', rotation='horizontal', ha='right')

    axi=0
    ax[axi, 1].set_xlabel('lag (ms)')
    ax[axi, 1].set_ylabel('XC (norm)')
    if norm_waveforms:
        ax[axi, 0].set_ylabel('Amplitude (norm)', rotation='vertical', ha='center')
        ax[axi, 0].set_ylim([-1.2,1.2])
    else:
        ax[axi, 0].set_ylabel('Amplitude (\u03bcV)')
    ax[axi, 0].set_xlabel('Time (ms)')
    ax[axi, 2].set_xlabel('Peak lag (ms)')
    ax[axi, 3].set_xlabel('XC at 0 lag')
    ax[axi, 1].set_ylim((-.5,.95))
    # Hide the right and top spines
    [ax_.spines.right.set_visible(False) for ax_ in ax[:, 0]]
    [ax_.spines.top.set_visible(False) for ax_ in ax[:,0]]
    [ax_.spines.bottom.set_visible(False) for ax_ in ax[1:, 0]]
    [ax_.set_yticklabels('') for ax_ in ax[:, 0]]
    # yl = ax[axi,0].get_ylim()
    # [ax_.set_ylim(yl) for ax_ in ax[:,0]];

    N = 3
    plot_XCpeak_pv=False
    if fit_XC0m is not None:
        N = N + 1
    if not plot_XCpeak_pv:
        N = N - 1
    ax2 = subfigs[1].subplots(N, 1)
    if plot_XCpeak_pv:
        ax2[0].plot(levels, XCpeak_pv, '.-')
        ii = [0, 1, 2, 3]
    else:
        ax2 = np.concatenate(([0],ax2))
        ii = [1, 2, 3]
    ax2[1].plot(levels, XCpeak_KSs, '.-', label='data')
    ax2[-1].plot(levels, p_near_0*100, '.-')
    if fit_KSs is not None:
        if type(fit_KSs) is dict:
            if fit_KSs['sigmoid_fit'] is not None:
                ax2[1].plot(levels, fit_KSs['sigmoid_fit']['yfit'], color=colors[1], label='sigmoid fit')
                ax2[1].plot(levels, fit_KSs['power_law_fit']['yfit'], color=colors[2], label='power-law fit')
            if fit_KSs['threshold'] is not None:
                ax2[1].text(ax2[1].get_xlim()[1], ax2[1].get_ylim()[0],
                            f"{(fit_KSs['bestFitType'] or 'No fit')}, thresh={fit_KSs['threshold']:.1f}",
                verticalalignment='bottom', horizontalalignment='right')
                ax2[1].axvline(fit_KSs['threshold'], color='r', linestyle='--', linewidth=.5)
        else:
            ax2[1].plot(levels, fit_KSs.yfit, color=colors[1], label='sigmoid fit')
        ax2[1].legend(loc='upper right', bbox_to_anchor=(1,.9))
    if fit_XCp_near_0 is not None:
        if type(fit_KSs) is dict:
            if fit_XCp_near_0['sigmoid_fit'] is not None:
                ax2[-1].plot(levels, fit_XCp_near_0['sigmoid_fit']['yfit']*100, color=colors[1], label='sigmoid fit')
                ax2[-1].plot(levels, fit_XCp_near_0['power_law_fit']['yfit']*100, color=colors[2], label='power-law fit')
            if fit_XCp_near_0['threshold'] is not None:
                ax2[-1].text(ax2[-1].get_xlim()[1], ax2[-1].get_ylim()[0],
                        f"{(fit_XCp_near_0['bestFitType'] or 'No fit')}, thresh={fit_XCp_near_0['threshold']:.1f}",
                verticalalignment='bottom', horizontalalignment='right')
                ax2[-1].axvline(fit_XCp_near_0['threshold'], color='r', linestyle='--', linewidth=.5)
        else:
            ax2[-1].plot(levels, fit_XCp_near_0.yfit*100, color=colors[1], label='sigmoid fit')
    ax2[1].axhline(.3, color='k', linestyle='--', linewidth=.5)
    ax2[1].text(ax2[1].get_xlim()[1], .31, 'criterion', horizontalalignment='right')
    ax2[-1].axhline(.4*100, color='k', linestyle='--', linewidth=.5)
    ax2[-1].text(ax2[-1].get_xlim()[1], 41, 'criterion', horizontalalignment='right')
    if plot_XCpeak_pv:
        ax2[0].set_ylabel('XC0_pval')
    ax2[1].set_ylabel('XC0_KSs')
    ax2[-1].set_ylabel('XCp_near_0')
    ax2[-1].set_xlabel('Level (dB SPL)')

    if fit_XC0m is not None:
        ax2[2].errorbar(levels, xc0.mean(axis=1), xc0.std(axis=1))
        ax2[2].set_ylim((-.3, 1.1))
        if fit_XC0m['sigmoid_fit'] is not None:
            ax2[2].plot(levels, fit_XC0m['sigmoid_fit']['yfit'], color=colors[1], label='sigmoid fit')
            ax2[2].plot(levels, fit_XC0m['power_law_fit']['yfit'], color=colors[2], label='power-law fit')
        if fit_XC0m['threshold'] is not None:
            ax2[2].text(ax2[2].get_xlim()[1], ax2[2].get_ylim()[0],
                f"{(fit_XC0m['bestFitType'] or 'No fit')}, thresh={fit_XC0m['threshold']:.1f}",
                verticalalignment='bottom', horizontalalignment='right')
            ax2[2].axvline(fit_XC0m['threshold'], color='r', linestyle='--', linewidth=.5)
        ax2[2].axhline(.3, color='k', linestyle='--', linewidth=.5)
        ax2[2].text(ax2[2].get_xlim()[1], .31, 'criterion', horizontalalignment='right')
        ax2[2].set_ylabel('XC0_mean')
        ax2[2].set_yticks((0,.2,.4,.6,.8,1))
        ax2[-1].set_xlabel('Level (dB SPL)')

    if human_threshold is not None:
        [ax_.axvline(human_threshold, color='k', linestyle='--', linewidth=.5) for ax_ in ax2[ii]]
    ax2[1].set_ylabel('KStest on XC at 0 lag')
    ax2[2].set_ylabel('Mean of XC at 0 lag')
    ax2[3].set_ylabel('% XC peaks near 0 lag')
    return fig_handle


thresholds = np.array([[71.2 , 76.61, 72.83, 67.58, 66.42, 74.1 , 72.25, 70.27, 72.4 ,
        71.83, 73.66, 70.39, 73.85, 71.42, 68.28, 75.25, 72.58, 71.13,
        72.07, 74.22, 71.08, 74.23, 71.96, 72.8 , 72.15, 70.59, 72.08,
        73.73, 72.8 , 67.95, 71.78, 66.76, 75.  , 69.3 , 76.7 , 73.82,
        74.94, 71.05, 72.67, 76.28, 70.76, 77.38, 74.62, 75.28, 75.31,
        71.95, 67.7 , 72.31, 70.88, 72.54],
       [70.85, 70.63, 72.02, 67.85, 67.21, 69.65, 71.05, 70.49, 69.19,
        68.03, 69.85, 68.98, 68.92, 74.77, 69.99, 72.01, 69.77, 71.03,
        68.87, 71.56, 66.36, 71.29, 71.88, 68.26, 69.24, 68.15, 69.19,
        67.23, 69.03, 70.53, 67.39, 68.92, 66.83, 69.75, 68.73, 68.73,
        69.79, 70.35, 70.89, 70.69, 69.23, 68.51, 70.08, 71.3 , 67.21,
        68.16, 71.98, 71.96, 69.88, 73.44],
       [68.55, 66.21, 69.79, 68.49, 67.07, 66.27, 67.91, 66.91, 65.85,
        66.7 , 67.48, 67.44, 67.12, 68.01, 68.13, 68.24, 71.39, 66.49,
        67.62, 65.82, 68.21, 67.93, 69.94, 71.19, 68.52, 69.08, 66.68,
        70.81, 66.97, 68.91, 69.25, 67.19, 69.95, 66.17, 69.69, 66.31,
        67.2 , 68.69, 69.37, 68.22, 64.62, 68.74, 67.86, 66.72, 70.06,
        68.9 , 68.06, 69.31, 67.64, 67.82],
       [66.38, 65.8 , 66.56, 66.22, 67.64, 67.79, 67.41, 66.26, 68.13,
        66.25, 65.39, 67.92, 67.28, 65.55, 65.38, 66.65, 68.14, 69.34,
        66.59, 66.56, 67.32, 65.72, 67.69, 67.76, 65.74, 68.81, 67.  ,
        68.15, 66.69, 67.47, 67.57, 66.18, 67.44, 66.22, 66.11, 66.39,
        67.04, 66.51, 66.41, 67.39, 67.03, 68.82, 68.9 , 67.82, 67.3 ,
        68.29, 66.52, 66.78, 67.62, 64.28],
       [65.67, 65.61, 66.13, 66.7 , 65.67, 65.47, 66.55, 65.93, 66.68,
        64.82, 66.73, 66.46, 67.6 , 66.35, 66.69, 66.61, 65.11, 65.89,
        66.63, 64.58, 66.77, 65.8 , 66.75, 67.08, 66.73, 65.95, 65.19,
        65.03, 65.07, 66.16, 66.68, 66.61, 65.94, 65.79, 66.39, 66.5 ,
        64.14, 67.04, 65.04, 63.58, 66.4 , 66.27, 65.44, 66.41, 66.02,
        64.51, 65.56, 67.2 , 66.19, 65.79],
       [64.74, 64.68, 65.3 , 64.03, 65.44, 65.25, 66.02, 64.86, 64.27,
        66.35, 66.14, 65.3 , 65.51, 65.04, 65.92, 65.39, 66.21, 65.37,
        65.57, 64.06, 65.41, 64.8 , 66.01, 66.35, 65.88, 66.09, 65.47,
        65.28, 66.03, 65.41, 65.08, 65.12, 65.47, 66.77, 65.57, 66.23,
        65.74, 65.53, 64.72, 65.94, 65.05, 66.18, 65.64, 66.45, 66.01,
        65.72, 65.58, 66.18, 66.58, 64.32],
       [65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12,
        65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12,
        65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12,
        65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12,
        65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12, 65.12,
        65.12, 65.12, 65.12, 65.12, 65.12]])

'''
Full:
1 need to re-do, starting at 22 to get high-th datasets
2 done semi-old?
3 just need 45
4 4kHz taking a very long time
5 on 16

Quick:
1-3 done ish
4,5 done

Std: error in 32.
'''