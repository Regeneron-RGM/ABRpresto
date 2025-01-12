import os
from pathlib import Path

import matplotlib.pyplot as plt

import ABRpresto
from ABRpresto import XCsub
from ABRpresto import utils
from ABRpresto.loader import LOADERS


def run_fit(path, loader, reprocess=False, XCsubargs=None, frequencies=None, fit_name='ABRpresto'):
    path = Path(path)
    if type(loader) is str:
        loader = LOADERS[loader]

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

    save_path = loader.get_save_path(path)
    print(f'Loading experiments from {path}')
    for freq, freq_df in loader.iter_experiments(path):

        if freq is not None:
            if frequencies is not None:
                if freq not in frequencies:
                    print(f"  skipping {freq:.0f} Hz")
                    continue
            fig_filename = save_path / f'{path.stem}_{fit_name}_fit {freq:.0f}.png'
            json_filename = save_path / f'{path.stem}_{fit_name}_fit {freq:.0f}.json'
            if not reprocess and fig_filename.exists() and json_filename.exists():
                print(f"  {freq:.0f} Hz already fit with {fit_name}")
                continue
            print(f"  processing {freq:.0f} Hz")
        else:
            fig_filename = save_path / f'{path.stem}_{fit_name}_fit.png'
            json_filename = save_path / f'{path.stem}_{fit_name}_fit.json'
            if not reprocess and fig_filename.exists() and json_filename.exists():
                print(f"  already fit with {fit_name}")
                continue
            print(f"  processing")

        # fit_results, figure = XCsub.estimate_threshold(freq_df, **XCsubargs)
        fit_results, figure = XCsub.estimate_threshold_by_Ntrials(freq_df, **XCsubargs)
        fit_results['ABRpresto version'] = ABRpresto.get_version()

        print(f"    threshold is {fit_results['threshold']:.1f}, fit with: {fit_results['status_message']}")

        # Save summary figure and fit results
        figure.savefig(fig_filename)
        utils.write_json(fit_results, json_filename)
        print(f"    exported fit results to {json_filename}")

        plt.close(figure)


def main_process():
    import argparse
    parser = argparse.ArgumentParser('abrpresto')
    parser.add_argument('paths', nargs='+', help='List of paths to process')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively iterate through all paths and runs ABRpresto on any psiexperiment ABR IO data found')
    parser.add_argument('--reprocess', action='store_true', help='Forces ABRpresto to reprocess data that has already been thresholded')
    parser.add_argument('--loader', choices=list(LOADERS.keys()), default='psi', help='File format to load')
    args = parser.parse_args()
    loader = LOADERS[args.loader]

    if args.recursive:
        for path in args.paths:
            i = None
            for i, exp_path in enumerate(loader.iter_path(path)):
                run_fit(exp_path, loader, args.reprocess)
            if i is None:
                print(f'No data found in {path} with {loader} format.')
    else:
        for path in args.paths:
            run_fit(path, loader, args.reprocess)


if __name__ == '__main__':
    main_process()
