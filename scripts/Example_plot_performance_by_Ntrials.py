import ABRpresto.utils
import os
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd


# Loads thresholds (either from fitted output .json files or save csv), generates summary statistics,
# and makes interactive plots

# Parameters
# Path to ABRpresto dataset (or another dataset you want to plot the performance of).
# pth = 'C:/Data/ABRpresto data/'  #If you download the full dataset, use this.
pth = os.path.realpath(f'../example_data_psi') + '/'  # use this to just plot the example data

#Change this to a different algorithm to load and compare other data
algorithm = 'ABRpresto_Ntrials'

# load_source = 'csv' #use this to load ABRpresto thrersholds from a csv file (like the one included with the dataset)
load_source = 'fitted json files' # use this to load ABRpresto files from the fitted json files. To do this you will first have to run ABRpresto on the dataset.


#Load algorithm thresholds
if load_source == 'fitted json files':
    df_ABRpresto = ABRpresto.utils.load_fits(pth, save=False, algorithm=algorithm)
elif load_source == 'csv':
    df_ABRpresto = pd.read_csv(pth + f'{algorithm} thresholds 10-29-24.csv')
else:
    raise RuntimeError("load_source must be 'fitted json files' or 'csv'")

### BRAD: NOT UPDATED BELOW HERE YET

df_ABRpresto.rename(columns={'threshold': f'{algorithm} threshold'}, inplace=True)

#load manual thresholds
df_Manual = pd.read_csv(pth+'Manual Thresholds.csv')

#merge algorithm and manual thresholds
df_merge = df_Manual.merge(df_ABRpresto, how='left',
                           on=['id', 'timepoint', 'frequency', 'ear']).sort_values(by=['id', 'ear', 'frequency'])

#Call function to impute infs, generate summary table, and make interactive plots
df, df_summary, pli = ABRpresto.utils.compare_thresholds(df_merge, thresholders=['manual', 'ABRpresto'])
