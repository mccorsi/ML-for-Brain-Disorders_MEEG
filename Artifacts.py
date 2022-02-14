##
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
if os.path.basename(os.getcwd()) != "ML-for-Brain-Disorders_MEEG":
    os.chdir("ML-for-Brain-Disorders_MEEG")
db_basedir = os.getcwd() + "/MNE_samples/"

path_figures_root=os.getcwd() + '/Figures/'
##
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(0, 60).load_data()  # just use a fraction of data for speed here
##
mag_channels = mne.pick_types(raw.info, meg='grad')
raw.plot(duration=30, order=mag_channels, n_channels=20,
         color=dict(mag='darkblue', grad='darkblue', eeg='darkblue', eog='k', ecg='m',
                    emg='k', ref_meg='steelblue', misc='k', stim='k',
                    resp='k', chpi='k'),
         remove_dc=True)
##
raw.pick(['grad']).load_data()
fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True)
plt.savefig(path_figures_root + "Section3_Artifacts_PowerLine.pdf", dpi=300)

##
# add some arrows at 60 Hz and its harmonics:
for ax in fig.axes[1:]:
    freqs = ax.lines[-1].get_xdata()
    psds = ax.lines[-1].get_ydata()
    for freq in (60, 120, 180, 240):
        idx = np.searchsorted(freqs, freq)
        ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
                 width=0.1, head_width=3, length_includes_head=True)
## https://mne.tools/dev/auto_tutorials/preprocessing/10_preprocessing_overview.html
##
mag_channels = mne.pick_types(raw.info, meg='mag')
raw.plot(duration=30, order=mag_channels, n_channels=20,
         remove_dc=True)
plt.savefig(path_figures_root + "Section3_Artifacts_Heart.pdf", dpi=300)
##
raw.pick(['eeg']).load_data()
raw.plot(duration=30, n_channels=20,
         remove_dc=True,
         color=dict(mag='darkblue', grad='darkblue', eeg='darkblue', eog='k', ecg='m',
              emg='k', ref_meg='steelblue', misc='k', stim='k',
              resp='k', chpi='k'),
         scalings=dict(mag=1e-12, grad=4e-11, eeg=99e-6, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4, whitened=1e2))
plt.savefig(path_figures_root + "Section3_Artifacts_Eyes.pdf", dpi=300)