"""
==============================================================
ML for Brain Disorders - MEEG - Section 3.1 - Examples of artifacts
===============================================================
This module is designed to generate interactive plots that show examples of artifacts

"""
# Author: Marie-Constance Corsi <marie.constance.corsi@gmail.com>

## import packages & set the current directory
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
if os.path.basename(os.getcwd()) != "ML-for-Brain-Disorders_MEEG":
    os.chdir("ML-for-Brain-Disorders_MEEG")
path_figures_root=os.getcwd() + '/Figures/'

## load data, from publicly available datasets in MNE

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(0, 60).load_data()  # just use a fraction of data for speed here

## eyes artifacts elicited via the eeg time series
raw_eeg=raw.copy()
raw_eeg.pick(['eeg']).load_data()
raw_eeg.plot(duration=30, n_channels=20,
         remove_dc=True,
         color=dict(eeg='darkblue'),
         scalings=dict(eeg=99e-6))
plt.savefig(path_figures_root + "Section3_Artifacts_Eyes.pdf", dpi=300)

## cardiac artifacts elicited via the magnetometers time series
raw_mag=raw.copy()
raw_mag.pick(['mag']).load_data()
raw_mag.plot(duration=30, n_channels=20,
         remove_dc=True)
plt.savefig(path_figures_root + "Section3_Artifacts_Heart.pdf", dpi=300)

## power line elicited via a plot of the power spectra - gradiometers
raw_grad=raw.copy()
raw_grad.pick(['grad']).load_data()
fig = raw_grad.plot_psd(tmax=np.inf, fmax=250, average=True)
plt.savefig(path_figures_root + "Section3_Artifacts_PowerLine.pdf", dpi=300)
