<h1 align="center">
    MST-m6A
    <br>
<h1>

<h4 align="center">Standalone program for the MST-m6A paper</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/Phlogistic-Rain/MST-m6A?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/Phlogistic-Rain/MST-m6A?" alt="forks"></a>
<a href="https://doi.org/10.5281/zenodo.12741567">
    <img src="https://zenodo.org/badge/doi/10.5281/zenodo.12741567.svg" alt="DOI">
</a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a>
</p>

## Introduction
This repository provides the standalone program for MST-m6A framework. The virtual environment, pre-trained weights, and final models are available via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12741567.svg)](https://doi.org/10.5281/zenodo.12741567)

## Installation
### Software requirements
* Windows 10
* NVIDIA GPU with at least 4GB of VRAM
* This source code has been already tested on RTX A5000, RTX 1660ti and RTX 4060

## Getting started
### Cloning this repository
```
git clone https://github.com/Phlogistic-Rain/MST-m6A.git
```
```
cd MST-m6A
```

### Downloading basline and final models
* Please download the virtual environment for Windows, pre-trained weights, and final models via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12741567.svg)](https://doi.org/10.5281/zenodo.12741567)
* For the virtual environment for Windows (m6A_env.zip), please extract it into the [m6A_env](https://github.com/Phlogistic-Rain/MST-m6A/tree/main/m6A_env) folder.
* For the pretrained_weights.zip file (DNABERT<sup>[1]</sup>), please extract it into the [pretrained_weights](https://github.com/Phlogistic-Rain/MST-m6A/tree/main/pretrained_weights) folder.
* For the final_models.zip file, please extract it into [final_models](https://github.com/Phlogistic-Rain/MST-m6A/tree/main/final_models) folder.

### Running prediction
#### Usage
* For Windows' users, directly run [infer.bat](https://github.com/Phlogistic-Rain/MST-m6A/tree/main/infer.bat) file.

## Citation
If you use this code or part of it, please cite the following papers:
```
@article{,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

## References
[1] Ji, Y., Zhou, Z., Liu, H., & Davuluri, R. V. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. <i>Bioinformatics</i>, 37(15), 2112-2120. <a href="https://doi.org/10.1093/bioinformatics/btab083"><img src="https://zenodo.org/badge/doi/10.1093/bioinformatics/btab083.svg" alt="DOI"></a>
