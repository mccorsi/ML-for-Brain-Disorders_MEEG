"""
==============================================================
ML for Brain Disorders - MEEG - Section 2 - M/EEG activity
===============================================================
This module is designed to generate interactive plots that show examples of M/EEG activity from synthetic signals

"""
# Author: Marie-Constance Corsi <marie.constance.corsi@gmail.com>

## import packages & set the current directory
import os
import mne
from mne.simulation import simulate_raw, add_noise
from mne.datasets import sample
from mne.time_frequency import fit_iir_model_raw
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, simulate_evoked
from moabb.paradigms import MotorImagery
import os.path as op
import numpy as np
from scipy.signal import unit_impulse
from matplotlib import pyplot as plt

from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

if os.path.basename(os.getcwd()) != "ML-for-Brain-Disorders_MEEG":
    os.chdir("ML-for-Brain-Disorders_MEEG")

path_figures_root=os.getcwd() + '/Figures/'
## Section 2.1.1 - simulated data - evoked responses
# adapted from the tutorial conceived by D. Strohmeier and A. Gramfort and available here: https://mne.tools/stable/auto_examples/simulation/simulate_evoked_data.html
def data_fun(times):
    """Function to generate random source time courses, to simulate P300"""
    return (-50e-9 * np.sin(30. * times) *
            np.exp(- (times - 0.30 + 0.05 * rng.randn(1)) ** 2 / 0.01))


plt.close('all')
# data as template:
data_path = sample.data_path()

raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg-proj.fif')
raw.add_proj(proj)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(ave_fname)

label_names = ['Vis-lh', 'Vis-rh']
labels = [mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
          for ln in label_names]

# Generation of source time courses from 2 dipoles:
times = np.arange(500, dtype=np.float64) / raw.info['sfreq'] - 0.1
rng = np.random.RandomState(42)

stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times,
                          random_state=42, labels=labels, data_fun=data_fun)

# Generation of noisy evoked data:
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
iir_filter = fit_iir_model_raw(raw, order=5, picks=picks, tmin=60, tmax=120)[1]
nave = 1000  # simulate average of 1000 epochs - not realistisc bur reduces noise
evoked = simulate_evoked(fwd, stc, info, cov, nave=nave, use_cps=True,
                         iir_filter=iir_filter)

# plot results:
colors = plt.cm.Set2(np.linspace(0, 1, 9))
plot_sparse_source_estimates(fwd['src'], stc, colors=colors, bgcolor=(1,1,1),
           linewidth=6, fontsize=15,
            opacity=0.1, high_resolution=True)
#evoked.plot_joint()
evoked.plot_joint(picks=['eeg'])
plt.savefig(path_figures_root + "Section2_EvokedResponses_EEG.pdf", dpi=300)

## Section 2.1.2 - oscillatory activity
# adapted from the tutorial conceived by M. van Vliet and available here:https://mne.tools/stable/auto_tutorials/simulation/80_dics.html

def coh_signal_gen(freq_oscil=10, std_fluc=0.1, t_rand=0.001,  n_times=100, modality="meg"):
    """Generate an oscillating signal.
        freq_oscil: frequency of the oscillations, in Hz
        std_fluc: standard deviation of the fluctuations added to the signal
        t_rand: variation in the instantaneous frequency of the signal
        n_times: number of samples to be generated
    Returns
    -------
    signal : ndarray
        The generated signal.
    """
    # Generate an oscillator with varying frequency and phase lag.
    signal = np.sin(2.0 * np.pi *
                    (freq_oscil * np.arange(n_times) / sfreq +
                     np.cumsum(t_rand * rand.randn(n_times))))

    # Add some random fluctuations to the signal.
    signal += std_fluc * rand.randn(n_times)

    # Scale the signal to be in the right order of magnitude (~100 nAm) for MEG data.
    if modality == "meg":
        signal *= 100e-9
    elif modality == "eeg":
        signal *= 100e-6

    return signal

plt.close('all')
# We use the MEG and MRI setup from the MNE-sample dataset
data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')

# Filenames for various files we'll be using
meg_path = op.join(data_path, 'MEG', 'sample')
raw_fname = op.join(meg_path, 'sample_audvis_raw.fif')
fwd_fname = op.join(meg_path, 'sample_audvis-meg-eeg-oct-6-fwd.fif')
cov_fname = op.join(meg_path, 'sample_audvis-cov.fif')
fwd = mne.read_forward_solution(fwd_fname)

# Seed for the random number generator
rand = np.random.RandomState(42)

# data simulation
sfreq = 50.  # Sampling frequency of the generated signal
n_samp = int(round(10. * sfreq))
times = np.arange(n_samp) / sfreq  # 10 seconds of signal
n_times = len(times)
channel_names = ['MEG 0522']
signal_osc = np.empty((1,500),dtype=object)

freqs=[2, 6, 10, 20, 40]
for f_id,f in enumerate(freqs):
    # simulation of 2 time series
    signal1 = coh_signal_gen(freq_oscil=f, std_fluc=0.3, t_rand=0.001,  n_times=n_times, modality= "meg")
    signal2 = coh_signal_gen(freq_oscil=f, std_fluc=0.3, t_rand=0.001,  n_times=n_times, modality= "meg")

    # The locations on the cortex where the signal will originate from. These
    # locations are indicated as vertex numbers.
    vertices = [[146374], [33830]]

    # Construct SourceEstimates that describe the signals at the cortical level.
    data = np.vstack((signal1, signal2))
    stc_signal = mne.SourceEstimate(
        data, vertices, tmin=0, tstep=1. / sfreq, subject='sample')
    stc_noise = stc_signal * 0.

    snr = 0.75  # Signal-to-noise ratio. Decrease to add more noise. default=1

    # simulation w/ grad here, can try with eeg or mag
    info = mne.io.read_raw(raw_fname).crop(0, 1).resample(50).info

    # Only use gradiometers
    picks = mne.pick_types(info, meg='grad', stim=True, exclude=())
    mne.pick_info(info, picks, copy=False)

    # Define a covariance matrix for the simulated noise. In this tutorial, we use a simple diagonal matrix.
    cov = mne.cov.make_ad_hoc_cov(info)
    cov['data'] *= (20. / snr) ** 2  # Scale the noise to achieve the desired SNR

    # Simulate the raw data, with a lowpass filter on the noise
    stcs = [(stc_signal, unit_impulse(n_samp, dtype=int) * 1),
            (stc_noise, unit_impulse(n_samp, dtype=int) * 2)]  # stacked in time
    duration = (len(stc_signal.times) * 2) / sfreq
    raw = simulate_raw(info, stcs, forward=fwd)
    add_noise(raw, cov, iir_filter=[4, -4, 0.8], random_state=rand)

    signal_osc=np.concatenate((signal_osc,raw[channel_names, 0:500][0]))

# plot results

plt.close('all')
signal_osc_plot=signal_osc
signal_osc_plot=signal_osc[1:len(freqs)+1,:]
time=raw[channel_names, 0:500][1]
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
colors = plt.cm.Set2(np.linspace(0, 1, 9))
for f_id in range(len(freqs)):
    y_offset = 9e-11*f_id
    y=signal_osc_plot[f_id,:].T + y_offset*np.ones((500)).T
    plt.plot(time, y, label=str(freqs[f_id]), color=colors[f_id,:])

ax.legend(labels=['delta','theta','alpha','beta','gamma'], bbox_to_anchor=(1.14, 1), frameon=False, prop={'size': 15})
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.ylabel('Amplitude (T or V)', fontsize=21)
plt.xlabel('Time (s)', fontsize=21)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(path_figures_root + "Section2_OscillatoryActivity.pdf", dpi=300)
