import os
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.viz import plot_raw
import matplotlib.pyplot as plt


from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

if os.path.basename(os.getcwd()) != "ML_Brain-disorders_MEEG":
    os.chdir("ML_Brain-disorders_MEEG")
db_basedir = os.getcwd() + "/MNE_samples/"

# TODO: easier to use a single dataset as "fil rouge" + ref to the paper & packages

#%% TODO: AQUI - Weibo 2014 : 7 (lh, rh, hands, feet, left_hand_right_foot, right_hand_left_foot, rest)
from moabb.datasets import Weibo2014

dataset=Weibo2014()
data = dataset.get_data(subjects=[1])
subject, session, run = 1, "session_0", "run_0"
raw = data[subject][session][run]

_ = raw.plot_sensors(show_names=True)

ch_eeg=['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2']
raw.info['bads']=['CB2','VEO','HEO']
raw.plot_psd(fmax=50, picks=ch_eeg, average=True)

_ = raw.plot(duration=4, n_channels=30, color={'eeg':'darkblue'})
_ = raw.plot_psd(fmin=4., fmax=35, picks=['eeg'])


from moabb.paradigms import LeftRightImagery
fmin, fmax= 1, 45
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)

X['left_hand'].plot()
X['right_hand'].plot()
# by default filter
X['left_hand'].plot_psd(picks='eeg', fmin=fmin, fmax=fmax)
X['right_hand'].plot_psd(picks='eeg', fmin=fmin, fmax=fmax)

X['right_hand'].plot_psd(picks=ch_eeg, fmin=fmin, fmax=fmax)
#%% eegbci dataset
# plot baseline, eyes closed to see alpha peak

#Define the parameters
subject = 1  # use data from subject 1
runs = [2]  # baseline eyes closed

#Get data and locate in to given path
files = eegbci.load_data(subject, runs, '../datasets/')
#Read raw data files where each file contains a run
raws_basel_cl = [read_raw_edf(f, preload=True) for f in files]
#Combine all loaded runs
raw_obj_basel_cl = concatenate_raws(raws_basel_cl)

raw_data_basel_cl = raw_obj_basel_cl.get_data()
print("Number of channels: ", str(len(raw_data_basel_cl)))
print("Number of samples: ", str(len(raw_data_basel_cl[0])))

#%% plots

raw_obj_basel_cl.plot(duration=120, n_channels=15, scalings=dict(eeg=420e-6))
raw_obj_basel_cl.plot_psd(average=True, fmax=50)