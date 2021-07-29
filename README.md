# Automated sleep stage classification of wide-field calcium imaging data via multiplex visibility graphs and deep learning

### Paper: TODO

**Authors:** Zhang X, Landsness EC, Chen W, Miao H, Tang M, Brier LM, Culver JC, Lee JM, Anastasio MA <br />
University of Illinois at Urbana-Champaign, Urbana, IL - 61801, USA <br>
Washington University School of Medicine<br>
Washington University School of Engineering<br>

**Contact:** xiaohui8@illinois.edu, maa@illinois.edu

**Abstract:** TODO
<p align="center">
<img src="./pic_recon/docs/face_recons_paper.png" alt="Face image reconstruction" width="500"/>
</p>

## System Requirements
- Linux/Unix-based systems recommended. The code hasn't been tested on Windows.
- Python 3.7.6. 
- Tensorflow 2.2.0.
- [Focal loss package](artemmavrin/focal-loss).
- SciPy, NumPy, scikit-image, sklearn, matplotlib
- [Fast natural visibility graph MATLAB package].(https://www.mathworks.com/matlabcentral/fileexchange/70432-fast-natural-visibility-graph-nvg-for-matlab) 
  
## Directory structure
The directory `network training` contains the top level scripts:
- `dataloader.py`: Script for loading the MVG representations from WFCI epochs
- `model_cnn2d.py`: Script for the compact 2D CNN to classify sleep
- `train.sh`: Scripts for running the network training.

The directory `MVG` contains the following sub-directories:

## Pretrained CNN weights
TODO

## Dataset
The WFCI data in this paper is available at: 

## References
TODO 





