# ML-for Brain Disorders_MEEG
---
This repository contains the code and supporting documents associated with the following manuscript:

https://hal.inria.fr/hal-03604421

Please cite as:

Corsi, M.-C. (2022). Electroencephalography and Magnetoencephalography. In: "Machine learning for brain diseases" edited by O. Colliot. In Press. Available in [HAL](https://hal.inria.fr/hal-03604421)


---
## Authors:
* [Marie-Constance Corsi](https://marieconstance-corsi.netlify.app), Postdoctoral Researcher, Aramis team-project, Inria Paris, Paris Brain Institute


---
## Abstract
In this chapter, we present the main characteristics of electroencephalography (EEG) and magnetoencephalography (MEG). More specifically, this chapter is dedicated to the presentation of the data, the way they can be acquired and analyzed. Then, we present the main features that can be extracted and their applications for brain disorders with concrete examples to illustrate them. Additional materials associated with this chapter are available in the dedicated Github repository.

---
## Data
All data associated with this manuscript are publicly available and can be found in the [Mother of all BCI Benchmarks (MOABB)](http://moabb.neurotechx.com/docs/index.html) here:
[http://moabb.neurotechx.com/docs/datasets.html](http://moabb.neurotechx.com/docs/datasets.html) and/or in [MNE-Python](https://mne.tools/stable/index.html)



## Code
This repository contains the code used to run the analysis performed and to plot the figures. Additional scripts aims at providing some examples of M/EEG analysis
To install all the packages used in this work you can directly type in your terminal:
`pip install -r requirements.txt`


The code proposed here relies on several open-source Python packages:

* [MNE-Python](https://mne.tools/stable/index.html) & [[Gramfort et al, 2014]](https://pubmed.ncbi.nlm.nih.gov/24161808/)
* [MOABB](http://moabb.neurotechx.com/docs/index.html) & [[Jayaram et al, 2018]](https://iopscience.iop.org/article/10.1088/1741-2552/aadea0)
* [scikit-learn](https://scikit-learn.org/stable/) & [[Pedregosa et al, 2011]](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html) & [[Buitinck et al, 2013]](https://hal.inria.fr/hal-00856511)
* [Matplotlib](https://matplotlib.org/stable/index.html)
* [Seaborn](https://seaborn.pydata.org) 
