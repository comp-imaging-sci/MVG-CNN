# Automated sleep stage classification of wide-field calcium imaging data via multiplex visibility graphs and deep learning

### Paper: TODO

**Authors:** Zhang X, Landsness EC, Chen W, Miao H, Tang M, Brier LM, Culver JC, Lee JM, Anastasio MA <br />
University of Illinois at Urbana-Champaign, Urbana, IL - 61801, USA <br>
Washington University School of Medicine<br>
Washington University School of Engineering<br>

**Contact:** xiaohui8@illinois.edu, maa@illinois.edu

**Abstract:** Wide-field calcium imaging (WFCI) allows for monitoring of cortex-wide neural dynamics in mice. When applied to the study of sleep, WFCI data are manually scored into the sleep states of wake, non-REM (NREM) and REM by use of adjunct EEG and EMG recordings. However, this process is time consuming and often suffers from low inter- and intra-rater reliability and invasiveness. Therefore, an automated sleep state classification method that operates on WFCI data alone is needed. A hybrid, two-step method is proposed. In the first step, spatial-temporal WFCI data is mapped to a multiplex visibility graph (MVG). Subsequently, a two-dimensional convolutional neural network (2D CNN) is employed on the MVGs to be classified as wake, NREM and REM.
<p align="center">
<img src="./MVG-CNN/Figure1.eps" alt="Schematic" width="500"/>
</p>

## System Requirements
- Linux/Unix-based systems recommended. The code hasn't been tested on Windows.
- Python 3.7.6. 
- Tensorflow 2.2.0.
- [Focal loss package](artemmavrin/focal-loss).
- SciPy, NumPy, scikit-image, sklearn, matplotlib
- [Fast natural visibility graph MATLAB package](https://www.mathworks.com/matlabcentral/fileexchange/70432-fast-natural-visibility-graph-nvg-for-matlab)
  
## Directory structure
The directory `network training` contains the top level scripts:
- `dataloader.py`: Script for loading the MVG representations from WFCI epochs
- `model_cnn2d.py`: Script for the compact 2D CNN to classify sleep
- `train.sh`: Scripts for running the network training.

The directory `MVG` contains the following sub-directories:

## Pretrained CNN weights
TODO

## Dataset
The WFCI data in this paper is available at PhysioNet: 

## References
TODO 





