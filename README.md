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
- 64 bit Python 3.6+. The code has been tested with Python 3.7.4 installed via Anaconda
- Tensorflow 1.14/1.15. The code has been tested with Tensorflow 1.14. Tensorflow 2+ is not supported.
- [imageio](https://imageio.readthedocs.io/en/stable/) - Install via `pip` or `conda`.
- [scikit-image](https://scikit-image.org/)

Additional dependencies that are required for the various reconstruction methods are as follows:
#### PLS-TV
- [Prox-TV](https://pythonhosted.org/prox_tv/)<br />
  Can be installed via `pip install prox_tv`. In our experience, this works on Linux and Mac, but not on Windows.
  
#### CSGM and PICGM
- [Cuda toolkit 10.0](https://developer.nvidia.com/cuda-toolkit) (higher versions may work but haven't been tested)
- GCC 7.2+. The code has been tested with GCC 7.2.0

#### PICCS
- [TF-Wavelets](https://github.com/UiO-CS/tf-wavelets): Relevant portions included with this code.
  
## Directory structure
The directory `scripts` contains the top level scripts:
- `get_network_weights.sh`: Script for downloading our pretrained StyleGAN2 network weights and storing them in `stylegan2/nets/`.
- `run_projector.sh`: Script for projecting an image onto the range of a given StyleGAN2.
- `recon_*.sh`: Scripts for running the various reconstruction algorithms.

The directory `stylegan2` contains the original StyleGAN2 code, with the following modifications:
- Addition of regularization based on Gaussianized disentangled latent-space for projecting images onto the range of the StyleGAN2 (contained in `stylegan2/projector.py` with regularization and optimization parameters controlled in `stylegan2/run_projector.py`). 
- `stylegan2/model_oop.py` - A small object-oriented wrapper around StyleGAN2.
- Various scripts for related to the theoretical results presented in our paper. 

`stylegan2/nets/` is the location for saving the trained network .pkl files.

The directory `pic_recon` contains the following sub-directories:
- `src` Stores all python codes for image reconstruction.
- `masks_mri` stores the MRI undersampling masks
- `ground_truths` - images of ground truths
- `prior_images`- Images and latent-projections of prior images.
- `results` - stores the results of the reconstruction.

## Pretrained StyleGAN2 weights
StyleGAN2 network weights, trained on brain MR images and FFHQ dataset images can be found here : https://databank.illinois.edu/datasets/IDB-4499850. They can be downloaded using `scripts/get_network_weights.sh`. More information can be found in `TRAINED_WEIGHTS_INFO.md`. 

## Projecting an image onto the latent space of StyleGAN2
1. Make sure `cuda-toolkit/10` and `gcc/7.2+` are loaded.
2. Download StyleGAN2 weights, for example <br /> ```bash scripts/get_network_weights.sh FastMRIT1T2```.
3. Run `bash scripts/run_projector.sh`. If you wish, you may set `network` inside `run_projector.sh` to the path to the StyleGAN2 network `.pkl` you want. The image size is 256x256x1 for brain images, and 128x128x3 for face images. The projected images along with their latent representations will be stored in `stylegan2/projected_images/`

## Performing image reconstructions:

The simplest way to get a recon algorithm `alg` is as follows:
1. Download the appropriate StyleGAN2 weights. For brain images, use <br /> ```bash scripts/get_network_weights.sh CompMRIT1T2```.<br /> For face images, use <br /> ```bash scripts/get_network_weights.sh FFHQ```. <br /> The weights are stored in `stylegan2/nets/`
3. Specify the correct network pkl path as `network_path` in `scripts/recon_${alg}.sh`. The correct one for brain images is already specified. (applies only to CSGM and PICGM).
4. Run <br />```bash scripts/recon_${alg}.sh``` <br> for an algorithm `alg`, where `alg` can be `plstv`, `csgm`, `piccs` or `picgm`.

### Further details:

The ground truth (GT) images can be found in `pic_recon/ground_truths/`. The prior images (PIs) can be found in `pic_recon/prior_images/`. The GTs and the PIs are organized according to `data_type`, which can be either `faces` or `brain`. 

For the brain images, we provide two example GT-PI pairs. Additional GT-PI pairs can be downloaded from [The Cancer Imaging Archive (TCIA) Brain-Tumor-Progression dataset](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression#3394811983c589667d0448b7be8e7831cbdcefa6). Links to the data use policy can be found in `pic_recon/ground_truths/brain/README.md`. 

The Shutterstock Collection containing the GT-PI pairs for the face images can be found here: https://www.shutterstock.com/collections/298591136-e81133c6. A Shutterstock license is needed to use these images. The preprocessing steps used  are described in our paper, and include manual cropping and resizing to 128x128x3. You may also use the [automatic face alignment procedure developed by V. Kazemi and J. Sullivan](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) for preprocessing, similar to the original FFHQ dataset. This is likely to improve the reconstruction performance of CSGM and PICGM further.

All the recon scripts contain the following common arguments:
- `process` : This is the index of the image to be reconstructed.
- `data_type` : Can be either `faces` or `brain`.
- `mask_type` : More generally refers to the forward model. For random Gaussian sensing, this argument takes the form `gaussian_${m_by_n}`. For simulated MRI with random cartesian undersampling, this argument takes the form `mask_rand_${n_by_m}x`. The Gaussian sensing matrices are generated in the code using a fixed seed. The undersampling masks for MRI are located inside `pic_recon/masks_mri/`. 
- `snr` : Measurement SNR.
- `gt_filename`: Path to the ground truth.
- `pi_filename`: (only for PICCS and PICGM) Path to the prior image.
- Various regularization parameters.

The results will be stored in `pic_recon/results/${data_type}/${process}/${algorithm}_${data_type}_${mask_type}_SNR${snr}/`.

## Citation
```
@article{kelkar2021prior,
  title={Prior Image-Constrained Reconstruction using Style-Based Generative Models},
  author={Kelkar, Varun A and Anastasio, Mark A},
  journal={arXiv preprint arXiv:2102.12525},
  year={2021}
}
```

## References
1. **FastMRI initiative database:** Zbontar, Jure, et al. "fastMRI: An open dataset and benchmarks for accelerated MRI." arXiv preprint arXiv:1811.08839 (2018).
2. **The Cancer Imaging Archive** Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057. https://doi.org/10.1007/s10278-013-9622-7
3. **TCIA Brain-Tumor-Progression** Schmainda KM, Prah M (2018). Data from Brain-Tumor-Progression. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2018.15quzvnb 
4. **FFHQ dataset** A Style-Based Generator Architecture for Generative Adversarial Networks
Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)
https://arxiv.org/abs/1812.04948, https://github.com/NVlabs/ffhq-dataset







