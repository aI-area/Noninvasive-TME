# Deep interpretable Radiogenomics (DiRG) Model Correlates MRI with TME for Noninvasive Cancer Subtyping

## Overview

![overview](figure1_overview.png)

## Getting Started

DiRG model uses deconvolution methods to obtain the cell type fraction and proportion from bulk data and constructs the relationship of MRI with cancer subtypes/microenvironment using on a deep radiogenomics model, while focusing on the interpretability of the modeling results.

### Prerequisites

Our modeling framework is based on PyTorch (https://pytorch.org).

#### Deconvolution
`BayesPrism` requires the following dependencies in R: `snowfall`, `NMF`, `gplots`, `scran`, `BiocParallel`, `Matrix`, and it can be installed as follows：
```
library("devtools");
install_github("Danko-Lab/BayesPrism/BayesPrism")
```
For more information about `BayesPrism` refers to https://www.nature.com/articles/s43018-022-00356-3 or https://github.com/Danko-Lab/BayesPrism

We also compared our model against `Bisque`, `CIBERSORT`, `xCell`, `quanTIseq`, and `EPIC`.

`xCell`, `quanTIseq`, `EPIC` were constructed based on the `immunedeconv` package (https://github.com/omnideconv/immunedeconv):
```
install.packages("remotes")
remotes::install_github("omnideconv/immunedeconv")
```
`CIBERSORT` can be installed using:
```
install.packages("devtools")
devtools::install_github("Moonerss/CIBERSORT")
```
`Bisque`  can be installed using:
```
install.packages("devtools")
devtools::install_github("cozygene/bisque")
```

#### Deep Radiogenomics
MRIs were converted to JPG/PNG using DICOM based on `pydicom`：  
```
pip install pydicom
```
Radiomics features extract by `pyradiomics`:
```
pip install pyradiomics
```

## Running the Model

### Model

- Run BayesPrism with `./Code/R/bayes_scATOMIC8_BRCA.Rmd`
- Run the other deconvolution methods:
  - Bisque：`./Code/R/bayes_bisque_BRCA.Rmd`
  - CIBERSORT：`./Code/R/Cibersort.R`
  - Other: `./Code/R/immunedeconv_all.R`
- Run and train our deep radiogenomics model with `./Code/train_model.py`
- Obtain the image marker for
  - Radiogenomics discovery and validation cohort (with gene): `./Code/Grad_CAM.py`
  - Radiomics cohort and clinical cohort (without gene): `./Code/validation_marker.py`
- Obtain important features of SMLP with `./Code/Important_feature.py`
- Extract radiomics feature with `./Code/radiomics_feature.py`
- Run our surrogate model (XGB) with `./Code/val_surrogate.ipynb`

### Analysis

- Run GSEA with `./Code/R/GSEA.R`
- Obtain heat map and PCA results in Figure 2 with `./Code/R/Heatmap_and_PCA.R`
- Obtain boxplot results in Figure 2 with `./Code/R/boxplot.R`


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* Huijun Li, Qiuxia Yang, Rui Zhang, Yize Mao, Xiaoli Li, Zequn Zhang, Yuxi Chen, Feng Zou, Chon Lok Lei, Peng Wang, and Hongyan Wu2

See also the list of [contributors](https://github.com/compbio-fhs/noninvasive-tme-detection/contributors) who participated in this project.

## License

This project is licensed under the BSD 3-Clause License, see [LICENSE](LICENSE) details.

## Acknowledging this work

If you publish any work based on the contents of this repository please cite:

TBU


