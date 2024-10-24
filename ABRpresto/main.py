from ABRpresto import XCsub
from ABRpresto import utils
import ABRpresto
import os
import pandas as pd
from pathlib import Path

from cftsdata import abr


def run_fit(path, reprocess=False, XCsubargs=None):
    path = Path(path)
    fig_filename = path / f'{path.stem}_XCsub_fit.png'
    json_filename = path / f'{path.stem}_XCsub_fit.json'

    if not reprocess and fig_filename.exists() and json_filename.exists():
        print(f"{path} already fitted with ABRpresto")
        return

    if XCsubargs is None:
        XCsubargs = {
            'seed': 0,
            'pst_range': [0.0005, 0.006],
            'N_shuffles': 500,
            'avmode': 'median',
            'peak_lag_threshold': 0.5,
            'XC0m_threshold': 0.3,
            'XC0m_sigbounds': 'increasing, midpoint within one step of x range',  # sets bounds to make slope positive,
            # and midpoint within [min(level) - step, max(level) + step] step is usually 5 dB
            'XC0m_plbounds': 'increasing',  # sets bounds to make slope positive
            'second_filter': 'pre-average',
            'calc_XC0m_only': True,
            'save_data_resamples': False  # use this to save intermediate data (from each resample)
        }

    print(f'Loading {path}')
    fh = abr.load(path)
    abr_single_trial_data = fh.get_epochs_filtered()

    print('Fitting with XCsub algorithm')
    fit_results, figure = XCsub.estimate_threshold(abr_single_trial_data, **XCsubargs)
    fit_results['ABRpresto version'] = ABRpresto.get_version()

    print(f"Threshold is {fit_results['threshold']:.1f}, fit with: {fit_results['status_message']}")

    # Save summary figure and fit results
    figure.savefig(fig_filename)
    utils.write_json(fit_results, json_filename)
    print(f"Exported fit results to {json_filename}")


def iter_psi(path):
    '''
    Iterates over all datasets generated by psiexperiment found in a folder
    '''
    for filename in Path(path).glob('**/*abr_io/erp_metadata.csv'):
        yield filename.parent


def main_process():
    import argparse
    parser = argparse.ArgumentParser('abrpresto')
    parser.add_argument('paths', nargs='+', help='List of paths to process')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively iterate through all paths and runs ABRpresto on any psiexperiment ABR IO data found')
    parser.add_argument('--reprocess', action='store_true', help='Forces ABRpresto to reprocess data that has already been thresholded')
    args = parser.parse_args()

    paths = [Path(path).absolute() for path in args.paths]
    for path in paths:
        if not path.is_dir():
            raise ValueError(f'{path} is not a directory')

    if args.recursive:
        for path in paths:
            for experiment in iter_psi(path):
                run_fit(experiment, args.reprocess)
    else:
        for path in paths:
            if not (path / 'erp_metadata.csv').exists():
                raise ValueError(f'{path} does not contain ABR data')
            run_fit(path, args.reprocess)


if __name__ == '__main__':
    main_process()
