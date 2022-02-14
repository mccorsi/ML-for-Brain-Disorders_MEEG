"""
==============================================================
ML for Brain Disorders - MEEG - Section 5 - Example of clinical application w/ BCI
===============================================================
This module is designed to propose an example of clinical application of BCI
Use of an open BCI dataset from [Scherer et al, 2015): https://doi.org/10.1371/journal.pone.0123727


Materials used to prepare this demo:
    - MOABB tutorial proposed during the 2021 vBCI meeting: https://github.com/lkorczowski/BCI-2021-Riemannian-Geometry-workshop/blob/master/notebooks/MOABB-approach.ipynb
"""
# Author: Marie-Constance Corsi <marie.constance.corsi@gmail.com>

## import packages & dataset
import os.path as osp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gzip
import warnings

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline


from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from moabb import set_log_level
from moabb.evaluations import WithinSessionEvaluation
import warnings

from moabb.datasets import (
    BNCI2015004,
)
from moabb.paradigms import MotorImagery, LeftRightImagery
from moabb.pipelines.csp import TRCSP

import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt

if os.path.basename(os.getcwd()) != "ML-for-Brain-Disorders_MEEG":
    os.chdir("ML-for-Brain-Disorders_MEEG")
path_figures_root=os.getcwd() + '/Figures/'

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
dataset = BNCI2015004()
print("There is {} subjects and {} sessions in this dataset".format(
    len(dataset.subject_list), dataset.n_sessions)
)

def _plot_rainclouds(
    df_results, hue_order, path_figures_root, yticks_figure, filename,palette='colorblind'):
    plt.style.use("dark_background")
    ort = "h"
    pal = palette
    sigma = 0.2
    dx = "pipeline"
    dy = "score"
    dhue = "pipeline"
    f, ax = plt.subplots(figsize=(12, 12))
    ax = pt.RainCloud(
        x=dx,
        y=dy,
        hue=dhue,
        hue_order=hue_order,
        order=hue_order,
        data=df_results,
        palette=pal,
        bw=sigma,
        width_viol=0.7,
        ax=ax,
        orient=ort,
        alpha=0.65,
        dodge=True,
        pointplot=True,
        move=0.2,
    )
    ax.get_legend().remove()
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.yticks(range(len(df_results["pipeline"].unique())),
               yticks_figure, fontsize=15, rotation=90)
    plt.xticks(fontsize=18)
    plt.ylabel("Pipeline", fontsize=18)
    plt.xlabel("Score", fontsize=18)
    plt.savefig(path_figures_root + filename + ".pdf", dpi=300)


## example with one patient A, only the first session for the moment
# TODO : show features: subplot power in a channel depending on the condition - 1h
data = dataset.get_data(subjects=[1])
subject, session, run = 1, "session_0", "run_0"
raw = data[subject][session][run]
datasets = [BNCI2015004()]

# montage
_ = raw.plot_sensors(show_names=True)
# raw time series
_ = raw.plot(duration=4, n_channels=12, color={'eeg':'darkblue'})
# plot power spectra
_ = raw.plot_psd(fmin=4., fmax=35, picks=['eeg'])


## here, we try to discriminate two states: "right-hand" motor imagery, and "feet" motor imagery
events = ["right_hand", "feet"]
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=8, fmax=35)

set_log_level("error")
warnings.filterwarnings("ignore")

datasets = [dataset]
# we compare different pipelines
    #  combination of the CSP spatial filtering with LDA
pipeline = Pipeline(steps=[('csp', CSP(n_components=8)),
                           ('lda', LDA())])
pipelines = {'csp+lda': pipeline}

    # use of CSP with elastic-net
parameters = {'l1_ratio': [0.2, 0.5, 0.8],
              'C': np.logspace(-1, 1, 3)}
elasticnet = GridSearchCV(LogisticRegression(penalty='elasticnet', solver='saga'),
                          parameters)

pipelines["csp+en"] = Pipeline(steps=[('csp', CSP(n_components=8)),
                                      ('en', elasticnet)])
    # elastic-net in the tangent space
pipelines["tgsp+en"] = Pipeline(steps=[('cov', Covariances("oas")),
                                       ('tg', TangentSpace(metric="riemann")),
                                       ('en', elasticnet)])


# we evaluate the classification performance for a given session - can take some time
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True
)
results = evaluation.process(pipelines)

## Plots
# Group level
_plot_rainclouds(
    df_results=results, hue_order=pipelines.keys(), path_figures_root=path_figures_root, yticks_figure=pipelines.keys(), filename='Section5_BCI_GroupAnalysis',palette='colorblind')

# Individual level
fig, ax = plt.subplots(figsize=(8, 7))
results["subj"] = results["subject"].apply(str)
sns.barplot(
    x="subj", y="score", hue="pipeline", data=results, orient="v", palette="colorblind", ax=ax
)
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.savefig(path_figures_root + 'Section5_BCI_IndividualAnalysis.pdf', dpi=300)
