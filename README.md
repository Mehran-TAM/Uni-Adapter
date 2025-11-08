# Uni-Adapter
This repository is the official implementation of the AAAI 2026 paper ["Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models"](http://arxiv.org/link).

## Overview
![](assets/architecture.png)

3D Vision-Language Foundation Models (VLFMs) have shown strong generalization and zero-shot recognition capabilities in open-world point cloud processing tasks. However, these models often underperform in practical scenarios where data are noisy, incomplete, or drawn from a different distribution than the training data. To address this, we propose **Uni-Adapter**, a novel training-free online test-time adaptation (TTA) strategy for 3D VLFMs based on dynamic prototype learning. We define a 3D cache to store class-specific cluster centers as prototypes, which are continuously updated to capture intra-class variability in heterogeneous data distributions. These dynamic prototypes serve as anchors for cache-based logit computation via similarity scoring. Simultaneously, a graph-based label smoothing module captures inter-prototype similarities to enforce label consistency among similar prototypes. Finally, we unify predictions from the original 3D VLFM and the refined 3D cache using entropy-weighted aggregation for reliable adaptation. Without retraining, Uni-Adapter effectively mitigates distribution shifts, achieving state-of-the-art performance on diverse 3D benchmarks over different 3D VLFMs—improving ModelNet-40C by \textbf{10.55%}, ScanObjectNN-C by \textbf{8.26%}, and ShapeNet-C by \textbf{4.49%} over the source 3D VLFMs.

## Motivation
![](assets/motivation.png)



## Environment
### Package Setup
* [dassl](https://github.com/auniquesun/dassl)
* Ubuntu 23.10
* Python 3.8.16
* PyTorch 1.12.0
* CUDA 11.6
* torchvision 0.13.0
* timm 0.9.16
* pueue & pueued 2.0.4

```sh
  # NOTE The option 1 is recommended. A complete package list is provided in `env.yaml`
  # option 1: create conda virtual env by your own
  conda create -n pointcache python=3.8.16
  codna activate pointcache
  # install torch
  pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
  # install dassl
  git clone https://github.com/auniquesun/dassl
  cd dassl/
  python setup.py develop # (no need to re-build if the source code is modified)

  # option 2: create conda virtual env according to the provided env.yaml
  conda env create -f env.yaml
  codna activate pointcache
```

`pueue` is a shell command management software, we use it for scheduling the model training & evaluation tasks, please refer to the [official page](https://github.com/Nukesor/pueue) for installation and basic usage. We recommend this tool because under its help you can run the experiments at scale thus save your time.

**NOTE:** We provide a complete package list of our virtual environment in `env.yaml`. Feel free to check whether you need a specific package. If it is the case, run the following command to install it, _e.g._ 
```sh
  pip install h5py==3.10.0 plyfile==1.0.3
```

### Pre-trained Weights
1. In the experiments, we use the following models as the baselines. The pre-trained weights of these models can be found in their public GitHub repositories. 
    * [ULIP-2](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip-2)
    * [OpenShape](https://github.com/Colin97/OpenShape_code/)
    * [Uni3D](https://github.com/baaivision/Uni3D)

    - **NOTE:** 
        1. ULIP-2 uses the same [text encoder](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip/image-text-encoder) as ULIP
        2. For OpenShape, we use the [pointbert-vitg14-rgb](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/tree/main) version
            - For text encoder in OpenShape, we use [CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) from **huggingface laion**
        3. For Uni3D, we use the [uni3d-g](https://huggingface.co/BAAI/Uni3D/tree/main/modelzoo/uni3d-g) version
            - For text encoder in Uni3D, we use [eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k](https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k) from **huggingface timm**

2. Make a folder called `weights` under this project and save the pre-trained weights into this folder. 

### Datasets
1. The folder structure of used datasets should be organized as follows.
```sh
    /path/to/Point-Cache
    |----data # placed in the same level as `runners`, `scripts`, etc. 
        |----modelnet_c
        |----modelnet40
        |----scanobjnn
        |----omniobject3d
            |----1024
        |----objaverse_lvis

    ...
```

2. You can find the download links of the above datasets as follows.
    - [Link](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/new-3ddg-benchmarks/xset/dg) for `omniobject3d`
    - [Link](https://huggingface.co/datasets/auniquesun/Point-Cache/tree/main) for `modelnet40`, `scanobjnn`, and `objaverse_lvis`

## Usage
Point-Cache is totally *training-free* and can operate in comparable efficiency with zero-shot inference of large multi-modal 3D models. Users can reproduce the results presented in the paper by directly inferring on the test datasets, as explained below. 

### Robustness evaluation on _ModelNet-C_
1. This part corresponds to the experiments in Section 4.2 (Table 1). 



## Acknowledgement
Our implementation is partially inspired by the following projects, thanks to their great work.

1. [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch)
2. [ULIP](https://github.com/salesforce/ULIP)
3. [OpenShape](https://github.com/ZrrSkywalker/PointCLIP)
4. [Uni3D](https://github.com/yangyangyang127/PointCLIP_V2)
5. [TDA](https://github.com/kdiAAA/TDA)

## Contact
If you have any question about our work, please search related issues or create a new one in this repository.
