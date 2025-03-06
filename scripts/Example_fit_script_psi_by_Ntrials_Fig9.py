import matplotlib
# matplotlib.use('QT5Agg')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
import os
from pathlib import Path

from cftsdata import abr

import ABRpresto
# plt.figure();plt.plot((1,3));plt.show()

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
    'save_data_resamples': False,  # use this to save intermediate data (from each resample)
    'fit_each_resample': True,
    'N_resamples': 50,
    'keep_fit_each_resample_fit_params': True,
    'paper_plot': True
}
fit_name = 'ABRpresto_Ntrials_paper'

print(os.path.basename(__file__))

for exi in range(2,3):
    filename = Path(f'../example_data_psi/Example_{exi+1} abr_io').absolute()
    print(f'Loading {filename}')

    # Load ABRdata from psiexperiment data file format
    fh = abr.load(filename)
    epochs = fh.get_epochs_filtered()

    # Loop through each of the frequencies in the file
    for freq, freq_epochs in epochs.groupby('frequency'):
        if freq !=4000:
            continue
        print(f'Fitting {freq} Hz with ABRpresto algorithm')

        fit_results, fig_handle = ABRpresto.XCsub.estimate_threshold_by_Ntrials(freq_epochs,
                                                                                figure_path = filename / f'{freq}Hz_{fit_name}_fit.png',
                                                                                **XCsubargs)
        fit_results['ABRpresto version'] = ABRpresto.get_version()

        # print(f"Threshold is {fit_results['threshold']:.1f}, fit with: {fit_results['status_message']}")

        # Save figure as png
        figname = filename / f'{freq}Hz_{fit_name}_fit.png'
        fig_handle.savefig(figname)
        print(f'Figure saved to {figname}')

        # Save fit results as json
        jsonname = filename / f'{freq}Hz_{fit_name}_fit.json'
        print(jsonname)
        ABRpresto.utils.write_json(fit_results, jsonname)
        print(f'Fit results saved to {jsonname}')
        plt.close('all')

    # In the left column the figures show mean +/- SE of all trials in black, and median (or mean, depending on AVmode) for
    # the two subsets. Waveforms are normalized (for each level all 3 lines are scaled by the peak-to-peak of the mean
    # of all trials). The right hand side shows mean correlation coefficient vs stimulus level. Sigmoid and power law fits
    # to this data are shown in green and purple. The threshold is shown by the pink dashed line.
