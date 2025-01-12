import os
import numpy as np
from scipy import signal
import json
import scipy.optimize as optimize
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from . import interactive_plots
from .loader import PSILoader
from PIL import Image

def crossCorr(x1, x2, norm=False):
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    if norm:
        x1 = x1 / (np.std(x1) * len(x1))
        x2 = x2 / np.std(x2)
    return np.correlate(x1, x2, 'same')


def fit_sigmoid_power_law(levels, y, thresholdCriterion, y_err=None, sigbounds=None, plbounds=None):
    """
        Fits sigmoid and power low functions to y, determines which to use based on mse, then computes threshold as
         where that fit crosses a criterion (default 0.3).

Parameters
    ----------
    levels: stimulus levels
    y: data to fit
    thresholdCriterion: The criterion value used to find threshold
    y_err: error on y. Sigmoid fit is weithed based on the sqrt(y_err).
       square root is used to de-emphasize errors. Otherwise the sloping part of the sigmoid isn't well-fit
    sigbounds:
        If 'increasing, midpoint within one step of x range' sets bounds to make slope positive,
                # and midpoint within [min(level) - step, max(level) + step]. step is usually 5 dB
    plbounds: string, default None
            If 'increasing' sets bounds to make slope positive

Outputs
    ----------
    fit_results: dictionary of fitting results, see XCsub.estimate_threshold for details
    threshold: the threshold estimated by the algorithm.

    """

    adjR2Criterion = 0.7

    thdEstimationFailed = True
    threshold = np.nan
    sigmoid_fit = None
    power_law_fit = None
    bestFitType = None

    # 1 check if cc passes criterion
    if min(y) > thresholdCriterion * 1.1:  # Used a factor of 1.1 to allow a little extrapolation
        threshold = -np.inf
        bestFitType = 'all above criterion, threshold is -inf'
        thdEstimationFailed = False
    elif max(y) < thresholdCriterion:
        threshold = np.inf
        bestFitType = 'all below criterion, threshold is inf'
        thdEstimationFailed = False
    else:
        power_law_fit = fit_power_law(levels, y, y_err, bounds=plbounds)
        sigmoid_fit = fit_sigmoid(levels, y, y_err, bounds=sigbounds)

        if (sigmoid_fit['yfit'][0] < thresholdCriterion) & (sigmoid_fit['yfit'][-1] > thresholdCriterion) & \
                (sigmoid_fit['sse'] < power_law_fit['sse']):
                threshold = sigmoid_get_threshold(thresholdCriterion, *sigmoid_fit['params'])
                bestFitType = 'sigmoid'
                thdEstimationFailed = False
        elif not((power_law_fit['yfit'][0] < thresholdCriterion) & (power_law_fit['yfit'][-1] > thresholdCriterion)):
            if (y > thresholdCriterion).sum() <= 2:
                # Cases where noise pushed some levels above criterion, but couldn't fit. Can safely call inf
                threshold = np.inf
                bestFitType = "most below criterion, but couldn't fit, threshold is inf"
                thdEstimationFailed = False
            elif (y < thresholdCriterion).sum() <= 2:
                # Cases where noise pushed some levels below criterion, but couldn't fit. Can safely call -inf
                threshold = -np.inf
                bestFitType = 'all above criterion, threshold is -inf'
                thdEstimationFailed = False
            # Otherwise default remain: threshold=nan and thdEstimationFailed=False.
            # We want to take a closer look at these cases
        else:
            threshold = power_law_get_threshold(thresholdCriterion, *power_law_fit['params'])
            if power_law_fit['adj_r2'] > adjR2Criterion:
                bestFitType = 'power law'
            else:
                bestFitType = 'power law (noisy)'
            thdEstimationFailed = False

    algpars = {'thresholdCriterion': thresholdCriterion, 'adjR2Criterion': adjR2Criterion}
    fit_results = {'threshold': threshold, 'sigmoid_fit': sigmoid_fit, 'power_law_fit': power_law_fit,
                   'bestFitType': bestFitType, 'thdEstimationFailed': thdEstimationFailed, 'algpars': algpars}
    return fit_results, threshold


def fit_sigmoid(x, y, y_err=None, bounds=None, calc_yfit=True):
    if bounds is None:
        bounds_ = (-np.inf, np.inf)
    elif bounds == 'increasing, midpoint within one step of x range':
        bounds_ = ((.1, 0, x.min() - np.diff(x[:2])[0], -0.5), (1, np.inf, x.max() + np.diff(x[:2])[0], 0.8))
    else:
        raise RuntimeError(f'bounds was set to "{bounds}" but this is not an option.')

    Pinit = sigmoid_find_initial_params(x, y, bounds=bounds, bounds_=bounds_)

    try:
        span = x.max() - x.min()
        # Grid 5 steps over range
        # P0_grid = (slice(Pinit[0], Pinit[0]+1, 1),slice(x.min(), x.max()+.00001, (span)/4))
        # Grid 3 steps from 1/6 to 5/6 range.
        P0_grid = (slice(Pinit[0], Pinit[0] + 1, 1),
                   slice(Pinit[1], Pinit[1] + 1, 1),
                   slice(x.min() + span / 8, x.max() - span / 8 + .0001, (span * 3 / 4) / 4),
                   slice(Pinit[3], Pinit[3] + 1, 1))
        # Quick grid search on x0 to get close to global minima
        Pinit = optimize.brute(sigmoid_obj_fn, P0_grid, args=(x, y), finish=None)
        # Unused code, was used when finish was fmin to ensure Pinit was in bounds
        # if (resbrute[0] > bounds_[0][0]) & (resbrute[0] < bounds_[1][0]) & \
        #         (resbrute[1] > bounds_[0][1]) & (resbrute[1] < bounds_[1][1]):
        #     Pinit = resbrute
        #     print(f'{Pinit}')
        # else:
        #     print(f'Grid search on {P0_grid} returned {resbrute}, which is outside the bounds {bounds_}. '
        #           f'Seeding curve_fit with default, {Pinit})')
        # least-squares on best grid fit
        if y_err is not None:
            # Take sqrt to de-emphasize errors. Otherwise the sloping part of the sigmoid isn't well-fit
            y_err = np.sqrt(y_err)
        P, pcov = optimize.curve_fit(
            sigmoid, x, y, p0=Pinit, method="trf", max_nfev=50000, sigma=y_err, bounds=bounds_
        )
        # print(f'[{Pinit[0]:.4f}, {Pinit[1]:.1f}] -> [{P[0]:.4f}, {P[1]:.1f}]')
    except RuntimeError:
        print(f'Fit failed, using default {Pinit}')
        P = Pinit

    if calc_yfit:
        yfit = sigmoid(x, *P)
        sse = np.sum((yfit - y)**2)
        return {'params': P, 'yfit': yfit, 'sse': sse}
    else:
        return P


def sigmoid_find_initial_params(x, y, bounds=None, bounds_=None):
    amplitude = np.max(y)
    x0 = np.mean(x)
    baseline = np.min(y)
    slope = 8 / amplitude * (amplitude - baseline) / (np.max(x) - np.min(x))
    if bounds == 'increasing, midpoint within one step of x range':
        if slope <= 0:
            raise RuntimeError("Negative initial slope. This shouldn't happen?")
        if baseline < bounds_[0][3]:
            baseline = bounds_[0][3]
        if baseline > bounds_[1][3]:
            baseline = bounds_[1][3]
    return np.array([amplitude, slope, x0, baseline])


def sigmoid(x, amplitude, slope, x0, baseline):
    y = amplitude / (1 + np.exp(-slope * (x - x0))) + baseline
    return y


def sigmoid_obj_fn(params, *data):
    yfit = sigmoid(data[0], *params)
    return np.sum((yfit - data[1]) ** 2)


def sigmoid_get_threshold(criterion, amplitude, slope, x0, baseline):
    if amplitude / (criterion - baseline) <= 1:
        return np.nan
    return x0 - np.log(amplitude / (criterion - baseline) - 1)/slope


def fit_power_law(x, y, y_err=None, bounds=None, calc_yfit=True):
    x = np.array(x)
    if np.min(x) == 0:
        offset = 1
    elif np.min(x) < 0:
        offset = -np.min(x)
    else:
        offset = 0

    xp = x + offset
    yp = y

    if bounds is None:
        bounds_ = (-np.inf, np.inf)
    elif bounds == 'increasing':
        bounds_ = ((-np.inf, 0, -np.inf), (np.inf, np.inf, np.inf))
    else:
        raise RuntimeError(f'bounds was set to "{bounds}" but this is not an option.')

    Pinit = power_law_find_initial_params(xp, yp, bounds=bounds)

    try:
        P, pcov = optimize.curve_fit(
            power_law, xp, yp, p0=Pinit, method="trf", max_nfev=50000, bounds=bounds_)

    except RuntimeError as RTE:
        print(f'Fit failed for power law with sigma, using default {Pinit}')
        print(RTE)
        P = Pinit
    except Exception as E:
        pass

    if calc_yfit:
        yfit = power_law(x, *P)
        sse = np.sum((yfit - y) ** 2)
        sstot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - sse / sstot

        n = len(y)
        free_params = len(P)
        adj_r2 = 1 - (((1 - r2) * (n - 1)) / (n - free_params - 1))
        return {'params': P, 'yfit': yfit, 'sse': sse, 'adj_r2': adj_r2}
    else:
        return P


def power_law_find_initial_params(x, y, bounds=None):
    idx = (x > 0) & (y > 0)
    if idx.sum() <= 2:
        power = 0.5
        if idx.sum() > 0:
            amplitude = (y[idx] / (x[idx] ** power)).mean()
        else:
            amplitude = 1
    else:
        x = x[idx]
        y = y[idx]
        if bounds == 'increasing':
            powers = np.log(y[0] / y[1:]) / np.log(x[0] / x[1:])
            if np.all(powers <= 0):
                power = 0.5
            else:
                power = np.mean(powers[powers > 0])
        else:
            power = np.mean(np.log(y[0] / y[1:]) / np.log(x[0] / x[1:]))
        amplitude = (y/(x**power)).mean()
    baseline = 0
    return np.array([amplitude, power, baseline])


def power_law(x, amp, power, baseline):
    return amp * x**power + baseline


def power_law_get_threshold(y,  amp, power, baseline):
    return ((y - baseline) / amp) ** (1 / power)


def filter(sig, fs, highpass, lowpass, order=1):
    if highpass >= lowpass:
        raise ValueError(f'Highpass must be < lowpass')
    Wn = highpass / (0.5 * fs), lowpass / (0.5 * fs)
    b, a = signal.iirfilter(order, Wn)
    return signal.filtfilt(b, a, sig, axis=-1)


def write_json(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f, cls=AutoThJsonEncoder, indent=4)


class AutoThJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif callable(obj):
            return f'{obj.__module__}.{obj.__name__}'
        else:
            return super().default(obj)


def Psi_to_csv_all(Psi_data_path, target_path, reprocess=False, load_options=None):
    path = Path(Psi_data_path)
    print(f'Converting all data in "{path}" to csv:')
    for i, exp_path in enumerate(PSILoader.iter_path(None, path)):
        Psi_to_csv(exp_path, target_path, reprocess=reprocess, load_options=load_options)


def Psi_to_csv(Psi_data_path, target_path, reprocess=False, load_options=None):
    os.makedirs(target_path, exist_ok=True)
    path = Path(Psi_data_path)
    print(f'Converting {path} to csv:')
    for freq, freq_df in PSILoader.iter_experiments(None, path, load_options):
        csv_path = Path(target_path) / f'{path.stem} {freq:.0f}.csv'
        if not reprocess and csv_path.exists():
            print(f"  {freq:.0f} Hz already exported to csv")
            continue
        freq_df.index = freq_df.index.droplevel('frequency')
        freq_df.to_csv(csv_path)
        print(f"    wrote {csv_path}")


def load_fits(pth, save=True, algorithm = 'ABRpresto'):
    """
            Loads fits by the algorithm specified stored in .json files, and returns dataframe. Optionally saves to csv.

    Parameters
        ----------
        pth: string. Path within which to search (recursively) for .json files to load
        algorithm: string, default ABRpresto. Algorithm to load data fromy: data to fit. Filenames are expected to be in the format:
             *_{algorithm}_fit*.json
        save: bool, default True. If True save loaded dataframe to csv in pth.

    Outputs
        ----------
        df: dataframe of fitted threshold and some metadata. Edit this function to load more metadata if wanted.

        """
    mouse_num = []
    timepoint = []
    ear = []
    frequency = []
    threshold = []
    status = []
    status_msg = []
    pths = []
    for filename in Path(pth).glob(f'**/*_{algorithm}_fit*.json'):
        if 'Copy' in str(filename):
            continue
        D = json.load(open(filename))
        threshold.append(D['threshold'])
        mouse_num.append(int(str(filename).split('\\')[-1].split('_')[0][5:]))
        timepoint.append(int(str(filename).split('\\')[-1].split('_')[1][9:]))
        ear.append(str(filename).split('\\')[-1].split('_')[2].split(' ')[0])
        frequency.append(int(str(filename).split(' ')[-1].split('.')[0]))
        status.append(D['status'])
        status_msg.append(D['status_message'])
        pths.append(filename.parent.joinpath(filename.name.replace('.json','.png')))

    df = pd.DataFrame(
        {'threshold': threshold, 'id': mouse_num, 'timepoint': timepoint, 'ear': ear, 'frequency': frequency,
         'status': status, 'status_message': status_msg, 'pth': pths})
    if save:
        save_pth = pth + f'{algorithm} thresholds.csv'
        df.to_csv(save_pth, index=False)
        print(f'Saved {len(df)} thresholds to {save_pth}')
    return df

def compare_thresholds(df, thresholders, impute_infs=True):
    """
            Creates scatter and histogram plots to compare one or more algorithms. If length of thresholders is 2,
            plots will be interactive. Clicking on a point will pop up the data from that point.

    Parameters
        ----------
        df: Dataframe of thresholds to compare
        thresholders: list of strings. List of thresholders to compare. Can be longer than2, but interactive plots only
                      work for length 2. Expects df to contain columns named
                      '{thresholders[0]} threshold', '{thresholders[1]} threshold', etc.
                      Thresholds will be compared with the first element in the list as reference.
        impute_infs: bool, default True. If True impute inf anf -inf in dataframe to 5 dB higher and less than
                     'min_level' and 'max_level', respectively. 'min_level' and 'max_level' should be columns of the
                    dataframe. They're only needed if impute_infs is True

    Outputs
        ----------
        df: dataframe used for plotting (after imputing if used)
        df_summary: table of summary statistics for each thresholder beyond the first (and relative to the first)
        pli: function handle of callback funtion run when clicking on 2d histogram. Used for debugging and data exploration.

        """

    if impute_infs:
        for thresholder in thresholders:
            colname = thresholder + ' threshold'
            ii = np.isinf(df[colname]) & (df[colname] < 0);
            df.loc[ii, colname] = df.loc[ii, 'min_level'] - 5
            ii = np.isinf(df[colname]) & (df[colname] > 0);
            df.loc[ii, colname] = df.loc[ii, 'max_level'] + 5

    df_summary = summarize_thd_diffs(df, thresholders[1:], reference_thresholder=thresholders[0], crits=(0, 5, 10))
    df['threshold diff'] = df[thresholders[1] + ' threshold'] - df[thresholders[0] + ' threshold']

    binsize = 5

    def pli(x_, y_, binsize=binsize):
        print(f'Got {x_},{y_}')
        if pli.x == x_ and pli.y == y_:
            pli.ind += 1
            print(f'Loaded, incrementing ind to {pli.ind}')
        else:
            pli.ind = 0
            pli.x = x_
            pli.y = y_
            if (x_ == 5) and (y_ == 0):
                pli.sel = df[thresholders[0] + ' threshold'].isnull() & df[(thresholders[1] + ' threshold')].isnull()
            elif (x_ == 5) and (y_ == 5):
                pli.sel = df[thresholders[0] + ' threshold'].isnull() & ~df[(thresholders[1] + ' threshold')].isnull()
            elif (x_ == 5) and (y_ == -5):
                pli.sel = ~df[thresholders[0] + ' threshold'].isnull() & df[(thresholders[1] + ' threshold')].isnull()
            else:
                pli.sel = (df['threshold diff'] >= (pli.y - binsize / 2)) & (
                        df['threshold diff'] < (pli.y + binsize / 2)) & \
                          (df[thresholders[0] + ' threshold'] >= (pli.x - binsize / 2)) & (
                                  df[thresholders[0] + ' threshold'] < (pli.x + binsize / 2))
            pli.pths = df[pli.sel]['pth'].values
            print(f'There are {pli.sel.sum()} points in this cell: {pli.pths}')
        print(f'**** {pli.pths[pli.ind]}')
        print(df[pli.sel].iloc[pli.ind])

        pth = pli.pths[pli.ind]
        img = Image.open(pth)
        print(pth)
        pli.axes.cla()
        pli.axes.imshow(img)
        pli.axes.axis('off')

    pli.ind = 0
    pli.sel = None
    pli.x = None
    pli.y = None
    pli.pths = None

    def pli_scatter(pth):
        img = Image.open(pth)
        print(pth)
        pli_scatter.axes.cla()
        pli_scatter.axes.imshow(img)
        pli_scatter.axes.axis('off')

    pli_scatter.ind = 0
    pli_scatter.sel = None
    pli_scatter.x = None
    pli_scatter.y = None
    pli_scatter.pths = None

    if len(thresholders) <= 2:
        pli.fig, pli.axes = plt.subplots(1, figsize=(5, 8),
                                         gridspec_kw={'top': 1, 'bottom': 0, 'left': 0, 'right': 1})
        pli_scatter.axes = pli.axes
    else:
        pli = None
        pli_scatter = None
    ax = plot_threshold_diffs(df, df_summary, thresholders[1:], reference_thresholder=thresholders[0], as_diff=True, norm='by_column', fn=pli)
    ax[-1, 0].set_xlabel(f'Median Threshold (dB SPL; {thresholders[0]})', fontsize=20)

    gs_kw = dict(hspace=0.1, wspace=0, left=0.08, right=.99, bottom=.1, top=.93)
    fig, ax = plt.subplots(1, len(thresholders)-1, figsize=(5 , 12), sharey='all', sharex='all',
                           gridspec_kw=gs_kw)
    if len(thresholders) == 2: ax = np.array(ax, ndmin=2)
    for i, thresholder in enumerate(thresholders[1:]):
        ax.flat[i].plot([10, 115], [10, 115], '--k')
        # ax.flat[i].scatter(df[thresholders[0] + ' threshold'], df[thresholder + ' threshold'], 3, 'k')
        interactive_plots.scatter(df[thresholders[0] + ' threshold'].values, df[thresholder + ' threshold'].values,
                                            df['pth'].values, ax=ax.flat[i], fn=pli_scatter, color='k')
        ax.flat[i].text(0, 0.99, f"spearman p = {df_summary.loc[thresholder, 'spearman corr']:.2f}",
                        transform=ax.flat[i].transAxes, va='top')
        ax.flat[i].set_title(thresholders[i])

    [ax_.set_xlim([7, 113]) for ax_ in ax.flat]
    [ax_.set_ylim([7, 113]) for ax_ in ax.flat]
    [ax_.set_aspect('equal', 'box') for ax_ in ax.flat]
    return df, df_summary, pli

def summarize_thd_diffs(df, thresholders, reference_thresholder, crits=(0, 5, 10), limits=None):
    """
                Creates a table of summary statistics for each thresholder relative to the reference_thresholder

        Parameters
            ----------
            df: Dataframe of thresholds to summarize
            thresholders: list of strings. List of thresholders to compare. Expects df to contain columns named
                          '{thresholders[0]} threshold', '{thresholders[1]} threshold', etc.
            reference_thresholder: Thresholds will be compared against this thresholder. Expects df to contain a column:
                         '{reference_thresholder} threshold'
            crits: tuple of floats, default (0,5,10). Criteria over which to quantify percent within +/- this value
            limits: None or tuple of floats, must be length 2, ex (10, 110). If not None, data outside these limits will
                    be set to these limits.

        Outputs
            ----------
            df_summary: table of summary statistics for each thresholder (relative to the reference_thresholder)

            """

    df_ = df.copy()
    DFS = pd.DataFrame(index=thresholders)
    DFS2 = pd.DataFrame(index=thresholders)
    for thresholder in thresholders:
        if limits is not None:
            df_.loc[(df_[thresholder + ' threshold'] < limits[0]), thresholder + ' threshold'] = limits[0]
            df_.loc[(df_[thresholder + ' threshold'] > limits[1]), thresholder + ' threshold'] = limits[1]
        df_['threshold diff'] = df_[thresholder + ' threshold'] - df_[reference_thresholder + ' threshold']
        # DFS.loc[user, 'Ntotal'] = int((~np.isnan(df_['threshold diff'])).sum())
        DFS2.loc[thresholder, 'MedianDiff'] = np.nanmedian(df_['threshold diff'])
        DFS.loc[thresholder, 'Ntotal'] = int((~np.isnan(df_[reference_thresholder + ' threshold'])).sum())
        crit_strs = []
        for crit in crits:
            if (type(crit) is str):
                if (crit[0] == 'p'):
                    DFS.loc[thresholder, f'N_below_{crit[1:]}dB'] = int((df_['threshold diff'] < -1*int(crit[1:])).sum())
                    DFS.loc[thresholder, f'N_above_{crit[1:]}dB'] = int((df_['threshold diff'] > 1 * int(crit[1:])).sum())
                    crit_strs.append(f'N_below_{crit[1:]}dB')
                    crit_strs.append(f'N_above_{crit[1:]}dB')
                else:
                    raise RuntimeError(f'Invalid crit {crit}')
            else:
                DFS.loc[thresholder, f'N_within_{crit}dB'] = int((np.abs(df_['threshold diff']) <= crit).sum())
                crit_strs.append(f'N_within_{crit}dB')
        DFS.loc[thresholder, f'N_outside crit'] = \
            int((np.abs(df_['threshold diff']) > np.array([crit for crit in crits if type(crit) is int]).max()).sum())
        DFS.loc[thresholder, 'N NaN'] = int((np.isnan(df_['threshold diff']) &
                                             ~np.isnan(df_[reference_thresholder + ' threshold'])).sum())

    DFS = DFS.astype(int)
    for thresholder in thresholders:
        DFS.loc[thresholder, 'spearman corr'] = \
            df_[[thresholder + ' threshold', reference_thresholder + ' threshold']].corr(method='spearman').values[0, 1]
    for crit in crit_strs:
        DFS[crit.replace('N_', 'perc_')] = DFS[crit] / DFS['Ntotal'] * 100
    df_summary = pd.concat((DFS, DFS2), axis=1)
    return df_summary

def plot_threshold_diffs(df, df_summary, thresholders, reference_thresholder, as_diff=False, ax=None, norm=None,
                         fn=None, limits=(10, 110)):
    """
            Creates 2d histogram plots to compare one or more algorithms. If length of thresholders is 1,
            plots will be interactive. Clicking on a square will pop up the data from that square. Clicking again in the
            same location will bring up more data from that square if it exists

    Parameters
        ----------
        df: Dataframe of thresholds to compare
        df_summary: table of summary statistics for each thresholder
        thresholders: list of strings. List of thresholders to compare. Expects df to contain columns named
                            '{thresholders[0]} threshold', '{thresholders[1]} threshold', etc.
        reference_thresholder: Thresholds will be compared against this thresholder. Expects df to contain a column:
                           '{reference_thresholder} threshold'
        as_diff: bool, default False. impute_infs: If True plot as difference histogram
        norm: one of {None, 'by_column'} If 'by_column', normalized within each column of the histogram
        ax: Axes in which to make plot. Is not passed axes will be created
        fn: Callback funtion to be called when clicking a square


    Outputs
        ----------
        df: dataframe used for plotting (after imputing if used)

        pli: function handle of callback funtion run when clicking on 2d histogram. Used for debugging and data exploration.

        """
    """
        Creates a table of summary statistics for each thresholder relative to the reference_thresholder

          Parameters
              ----------
              df: Dataframe of thresholds to summarize
              thresholders: list of strings. List of thresholders to compare. Expects df to contain columns named
                            '{thresholders[0]} threshold', '{thresholders[1]} threshold', etc.
              reference_thresholder: Thresholds will be compared against this thresholder. Expects df to contain a column:
                           '{reference_thresholder} threshold'
              crits: tuple of floats, default (0,5,10). Criteria over which to quantify percent within +/- this value
              limits: Tuple of floats, default (10, 110). Data will be limited to these values. Vertical lines inside
                       will be drawn to indicate values that were infinite and were imputer.


              """

    percs = df_summary.keys()[df_summary.keys().str.contains('perc_within')].values
    percs_num = [p.split('_')[2][:-2] for p in percs]
    min_val = 10
    max_val = 110
    bin_step = 5
    xbins = np.arange(min_val - bin_step, max_val + 2 * bin_step, bin_step) - bin_step / 2
    # print(xbins)
    if norm == 'by_column':
        vmin = 0
        vmax = 1
    else:
        vmin = None
        vmax = None
    if as_diff:
        ybins = np.arange(-42.5, 42.5 + bin_step, bin_step) - bin_step / 2
        # ybins = np.append(np.append(np.arange(-50,0,bin_step),[-1,1]),np.arange(5,50+bin_step,bin_step))
    else:
        ybins = xbins
    make_axes = ax is None
    if make_axes:
        fig, ax = plt.subplots(1, len(thresholders), figsize=(12,10))
        ax = np.array(ax, ndmin=2)
    if len(thresholders) == 1:
        fs = 20
        ls = 14
    else:
        fs = 12
        ls = 12
    axf = ax.flatten()
    for n, thresholder in enumerate(thresholders):
        y = df[thresholder + ' threshold'].copy()
        y.loc[y > limits[1]] = limits[1]
        y.loc[y < limits[0]] = limits[0]
        ylabel = 'Individual Threshold (dB SPL)'
        if as_diff:
            y = y - df[reference_thresholder + ' threshold']
            ylabel = 'Difference (Individual - Median)'
        H, xedges, yedges = np.histogram2d(df[reference_thresholder + ' threshold'], y, bins=(xbins, ybins))
        # X, Y = np.meshgrid(xedges, yedges)
        if norm == 'by_column':
            H = (H.T / H.sum(axis=1)).T

        # interactive_plots.pcolor(X, Y, H.T, cmap='gray_r', vmin=vmin, vmax=vmax, ax=axf[n])
        ph = interactive_plots.pcolor(xedges, yedges, H.T, cmap='gray_r', vmin=vmin, vmax=vmax, ax=axf[n], mesh=False, fn=fn)
        cbh = plt.colorbar(ph)
        cbh.set_ticks((0, .5, 1), labels=('0', '50', '100'))
        cbh.set_label(label='Percent of Occurrences', fontsize=20)
        if as_diff:
            y_line = 10 # 7.5
            axf[n].plot(xbins[[0, -1]], (-1*y_line, -1*y_line), '--k', linewidth=1)
            axf[n].plot(xbins[[0, -1]], (y_line, y_line), '--k', linewidth=1)

        axf[n].set_title(f'{thresholder}: [{df_summary.loc[thresholder, percs[0]]:.0f},  {df_summary.loc[thresholder, percs[1]]:.0f},'
                         f'  {df_summary.loc[thresholder, percs[2]]:.0f}]%, N={df_summary.loc[thresholder, "Ntotal"]}, Nnan={df_summary.loc[thresholder,"N NaN"]}', fontsize=fs)
        axf[n].set_aspect('equal', 'box')
        axf[n].axvline(limits[0] + bin_step/2, ls='--', color='darkred', linewidth=1)
        axf[n].axvline(limits[1] - bin_step/2, ls='--', color='darkred', linewidth=1)

    [ax_.tick_params(axis='both', labelsize=ls) for ax_ in axf]
    if make_axes:
        ax[-1, 0].set_xlabel('Median Threshold (dB SPL)', fontsize=fs)
        ax[-1, 0].set_ylabel(ylabel, fontsize=fs)
    thresholder = thresholders[0]
    if make_axes:
        axf[0].set_title(
            f'   [0,  $\\pm{percs_num[1]}$, $\\pm{percs_num[2]}$] dB\n{thresholder}: [{df_summary.loc[thresholder, percs[0]]:.0f},  {df_summary.loc[thresholder, percs[1]]:.0f},'
            f'  {df_summary.loc[thresholder, percs[2]]:.0f}]%, N={df_summary.loc[thresholder, "Ntotal"]}, Nnan={df_summary.loc[thresholder,"N NaN"]}', fontsize=fs)
    return ax