# Uni-Adapter
This repository is the official implementation of the AAAI 2026 paper ["Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models"](http://arxiv.org/link).

🚧 **Code Release Notice**

The code and pretrained checkpoints are currently being prepared for public release.  
Please stay tuned — the full implementation will be made available **soon**.


## Overview

3D Vision-Language Foundation Models (VLFMs) have shown strong generalization and zero-shot recognition capabilities in open-world point cloud processing tasks. However, these models often underperform in practical scenarios where data are noisy, incomplete, or drawn from a different distribution than the training data. To address this, we propose **Uni-Adapter**, a novel training-free online test-time adaptation (TTA) strategy for 3D VLFMs based on dynamic prototype learning. We define a 3D cache to store class-specific cluster centers as prototypes, which are continuously updated to capture intra-class variability in heterogeneous data distributions. These dynamic prototypes serve as anchors for cache-based logit computation via similarity scoring. Simultaneously, a graph-based label smoothing module captures inter-prototype similarities to enforce label consistency among similar prototypes. Finally, we unify predictions from the original 3D VLFM and the refined 3D cache using entropy-weighted aggregation for reliable adaptation. Without retraining, Uni-Adapter effectively mitigates distribution shifts, achieving state-of-the-art performance on diverse 3D benchmarks over different 3D VLFMs—improving ModelNet-40C by 10.55%, ScanObjectNN-C by 8.26%, and ShapeNet-C by 4.49% over the source 3D VLFMs.

## Motivation

Existing test-time adaptation methods either depend on computationally expensive parameter updates or rely on high-confidence samples that fail to capture the full diversity of 3D structures. To address these limitations, this work introduces a training-free, prototype-based adaptation framework designed to enhance reliability and stability under challenging and heterogeneous test conditions.

## Environment


### Package Setup
* Ubuntu 23.10
* Python 3.8.16
* PyTorch 1.12.0
* CUDA 11.6
* torchvision 0.13.0

```sh
# NOTE: Option 1 is recommended. A complete package list is available in `env.yaml`.

# Option 1: Manually create a Conda virtual environment
conda create -n uniadapter python=3.8.16
conda activate uniadapter

# Install PyTorch
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install Dassl
git clone https://github.com/...
cd dassl/
python setup.py develop  # No need to rebuild if the source code is modified.

# Option 2: Create the Conda environment from the provided env.yaml
conda env create -f env.yaml

```


### Pre-trained Weights

We use the following models as baselines. Their pre-trained weights can be obtained from the respective public repositories:

* [ULIP-2](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip-2)  
* [OpenShape](https://github.com/Colin97/OpenShape_code/)  
* [Uni3D](https://github.com/baaivision/Uni3D)  

**Details on the models and text encoders used:**

1. **ULIP-2**  
   - Model: Use the same [text encoder](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip/image-text-encoder).  

2. **OpenShape**  
   - Model: [pointbert-vitg14-rgb](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/tree/main)  
   - Text Encoder: [CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)  

3. **Uni3D**  
   - Model: [uni3d-L](https://huggingface.co/BAAI/Uni3D/tree/main/modelzoo/uni3d-L)  
   - Text Encoder: [eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k](https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k)



### Datasets
1. The folder structure of used datasets should be organized as follows.
```sh
    
  |----data   
      |----modelnet40_c
      |----shapenet_c
      |----scanobjnn_c
      |----omniobject3d
          |----1024
      |----objaverse_lvis


    ...
```

### Evaluation on _ModelNet-C_

For ModelNet40-C, we used the corrupted data from [Modelnet40-C](https://github.com/jiachens/ModelNet40-C) repository, using the maximum corruption severity level (5) for all experiments



## Acknowledgement
Our implementation is partially inspired by the following projects, thanks to their great work.

1. [Point-Cache](https://github.com/auniquesun/Point-Cache)
2. [ULIP](https://github.com/salesforce/ULIP)
3. [OpenShape](https://github.com/ZrrSkywalker/PointCLIP)
4. [Uni3D](https://github.com/yangyangyang127/PointCLIP_V2)
5. [TDA](https://github.com/kdiAAA/TDA)

## Contact
If you have any question about our work, please search related issues or create a new one in this repository.
