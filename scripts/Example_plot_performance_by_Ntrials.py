import ABRpresto.utils
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Loads thresholds (either from fitted output .json files or save csv), generates summary statistics,
# and makes interactive plots

# Parameters
# Path to ABRpresto dataset (or another dataset you want to plot the performance of).
# pth = 'C:/Data/ABRpresto data/'  #If you download the full dataset, use this.
pth = os.path.realpath(f'../example_data_psi') + '/'  # use this to just plot the example data

#Change this to a different algorithm to load and compare other data
algorithm = 'ABRpresto_Ntrials_std'

# load_source = 'csv' #use this to load ABRpresto thrersholds from a csv file (like the one included with the dataset)
load_source = 'fitted json files' # use this to load ABRpresto files from the fitted json files. To do this you will first have to run ABRpresto on the dataset.


#Load algorithm thresholds
if load_source == 'fitted json files':
    df_ABRpresto = ABRpresto.utils.load_fits_Ntrials(pth, save=False, algorithm=algorithm)
elif load_source == 'csv':
    df_ABRpresto = pd.read_csv(pth + f'{algorithm} thresholds 10-29-24.csv')
else:
    raise RuntimeError("load_source must be 'fitted json files' or 'csv'")

N_trials_one_polarity=256*.125*np.arange(2,9)


import colorcet as cc


df = df_ABRpresto[df_ABRpresto.apply(lambda x: x.threshold[-1,-1] < 75, axis=1)].reset_index()
col = cc.glasbey[:len(df)]
rn = np.random.RandomState(0)
f,ax = plt.subplots(3,2, gridspec_kw={'wspace':0,'hspace':0}, figsize=(18,12))
for i,row in df.iterrows():
    ax[0,0].errorbar(N_trials_one_polarity + 10*(rn.rand(1)-.5), row['threshold'].mean(axis=1) - row['threshold'][-1,-1],
             row['threshold'].std(axis=1), label=str(row['pth']).split('\\')[-2].split(' ')[0].replace('_','').replace('Example','Ex') + ' ' + str(row['pth']).split('\\')[-1].split('_')[0],
                     color=col[i])
    ax[0,1].errorbar(row['noise']*1e9/np.sqrt(N_trials_one_polarity/N_trials_one_polarity[-1]), row['threshold'].mean(axis=1) - row['threshold'][-1,-1],
             row['threshold'].std(axis=1), color=col[i])

rn = np.random.RandomState(0)
v = 'thresholds_by_resample_mean'
for i,row in df.iterrows():
    ax[1,0].errorbar(N_trials_one_polarity + 10*(rn.rand(1)-.5), row[v].mean(axis=1) - row[v][-1,-1],
             row[v].std(axis=1), color=col[i])
    ax[1,1].errorbar(row['noise']*1e9/np.sqrt(N_trials_one_polarity/N_trials_one_polarity[-1]), row[v].mean(axis=1) - row[v][-1,-1],
             row[v].std(axis=1), color=col[i])

v = 'thresholds_by_resample_std'
for i,row in df.iterrows():
    ax[2,0].errorbar(N_trials_one_polarity + 10*(rn.rand(1)-.5), row[v].mean(axis=1) ,
             row[v].std(axis=1), color=col[i])
    ax[2,1].errorbar(row['noise']*1e9/np.sqrt(N_trials_one_polarity/N_trials_one_polarity[-1]), row[v].mean(axis=1),
             row[v].std(axis=1), color=col[i])

ax[0,0].legend(fontsize=6, ncol=2)
ax[0,0].set_ylabel('Threshold Difference (dB)\nAll resamples in one fit')
ax[1,0].set_ylabel('Threshold Difference (dB)\nMean across separate fits per resample')
ax[2,0].set_ylabel('Threshold Std (dB)\nStd across separate fits per resample')
[ax_.set_yticklabels('') for ax_ in ax[:,1]]
ax[2,0].set_xlabel('N resamples')
ax[2,0].set_xlabel('N trials per subaverage')
ax[2,1].set_xlabel('Noise level (nV)')
f.savefig(r'C:\Users\LukeS\Documents\Code\ABRpresto\Ntrials_plots\Thresholds_by_Ntrials_Example_set_difference.png')



df = df_ABRpresto[df_ABRpresto.apply(lambda x: x.threshold[-1,-1] < 75, axis=1)].reset_index()
col = cc.glasbey[:len(df)]
rn = np.random.RandomState(0)
f,ax = plt.subplots(3,2, gridspec_kw={'wspace':0,'hspace':0}, figsize=(18,12))
for i,row in df.iterrows():
    ax[0,0].errorbar(N_trials_one_polarity + 10*(rn.rand(1)-.5), row['threshold'].mean(axis=1),
             row['threshold'].std(axis=1), label=str(row['pth']).split('\\')[-2].split(' ')[0].replace('_','').replace('Example','Ex') + ' ' + str(row['pth']).split('\\')[-1].split('_')[0],
                     color=col[i])
    ax[0,1].errorbar(row['noise']*1e9/np.sqrt(N_trials_one_polarity/N_trials_one_polarity[-1]), row['threshold'].mean(axis=1),
             row['threshold'].std(axis=1), color=col[i])

rn = np.random.RandomState(0)
v = 'thresholds_by_resample_mean'
for i,row in df.iterrows():
    ax[1,0].errorbar(N_trials_one_polarity + 10*(rn.rand(1)-.5), row[v].mean(axis=1),
             row[v].std(axis=1), color=col[i])
    ax[1,1].errorbar(row['noise']*1e9/np.sqrt(N_trials_one_polarity/N_trials_one_polarity[-1]), row[v].mean(axis=1),
             row[v].std(axis=1), color=col[i])

v = 'thresholds_by_resample_std'
for i,row in df.iterrows():
    ax[2,0].errorbar(N_trials_one_polarity + 10*(rn.rand(1)-.5), row[v].mean(axis=1) ,
             row[v].std(axis=1), color=col[i])
    ax[2,1].errorbar(row['noise']*1e9/np.sqrt(N_trials_one_polarity/N_trials_one_polarity[-1]), row[v].mean(axis=1),
             row[v].std(axis=1), color=col[i])

ax[0,0].legend(fontsize=6, ncol=2)
ax[0,0].set_ylabel('Threshold (dB SPL)\nAll resamples in one fit')
ax[1,0].set_ylabel('Threshold (dB SPL)\nMean across separate fits per resample')
ax[2,0].set_ylabel('Threshold Std (dB)\nStd across separate fits per resample')
[ax_.set_yticklabels('') for ax_ in ax[:,1]]
ax[2,0].set_xlabel('N resamples')
ax[2,0].set_xlabel('N trials per subaverage')
ax[2,1].set_xlabel('Noise level (nV)')
f.savefig(r'C:\Users\LukeS\Documents\Code\ABRpresto\Ntrials_plots\Thresholds_by_Ntrials_Example_set.png')