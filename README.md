<div align="center">
 
# SAS: Segment Any 3D Scene with Integrated 2D Priors
![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://arxiv.org/abs/2503.08512'><img src='https://img.shields.io/badge/arXiv-2503.08512-b31b1b.svg'></a> &nbsp;
 <a href='https://peoplelu.github.io/SAS.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;

<a href="https://openreview.net/profile?id=~Zhuoyuan_Li4">Zhuoyuan Li</a><sup>1*</sup>,</span>
<a href="https://scholar.google.com/citations?user=cRpteW4AAAAJ&hl=zh-CN">Jiahao Lu</a><sup>1*</sup>,</span>
<a href="https://scholar.google.com/citations?user=-0y0FpkAAAAJ&hl=zh-CN">Jiacheng Deng</a><sup>1</sup>,
<a href="">Hanzhi Chang</a><sup>1</sup>,
<a href="">Lifan Wu</a><sup>1</sup>,
<a href="https://github.com/Rosetta-Leong">Yanzhe Liang</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=9sCGe-gAAAAJ&hl=zh-CN">Tianzhu Zhang</a><sup>1&dagger;</sup>

<sup>1</sup>University of Science and Technology &nbsp;&nbsp;

<sup>*</sup>Equal contribution
<sup>&dagger;</sup>Corresponding author

![teaser](teaser_00.jpg)
</div>

## :rocket: News

**11/Mar/2025**: We release our paper to [Arxiv](https://arxiv.org/abs/2503.08512).


<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#model-capability-construction">Model Capability Construction</a>
    </li>
    <li>
      <a href="#Train">Train</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## Installation
Start by cloning the repo:
```bash
git clone https://github.com/peoplelu/SAS.git
cd SAS
```

For linux, you need to install `libopenexr-dev` before creating the environment.
```bash
sudo apt-get install libopenexr-dev
conda create -n SAS python=3.8
conda activate SAS
```

Step 1: install PyTorch (We tested on pytorch 2.1.0 and cuda 11.8. Other versions may also work.):

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Step 2: install MinkowskiNet:

```bash
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
                           --install-option="--blas=openblas"
```

Step 3: install scatter for superpoint operation:
```bash
pip install torch-scatter
```

Step 4: install the remaining dependencies:
```bash
pip install scipy, open3d, ftfy, tensorboardx, tqdm, imageio, plyfile, opencv-python, sharedarray
pip install git+https://github.com/openai/CLIP.git
```

Step 5: install tensorflow:
```bash
pip install tensorflow==2.13.1
```

Step 6: Install SAM
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Step 7: Install LSeg and SEEM

Please create another two environments, lseg and seem, to install dependencies for [LSeg](https://github.com/isl-org/lang-seg) and [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once). You can refer to their official repo for details.

Step 8: Install dependencies for Stable Diffusion
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```




## Data Preparation
### Download pre-processed data
We provide the pre-processed point features from LSeg and SEEM, fused point features, and the constructed capabilities for the following datasets in [hugging face](https://huggingface.co/datasets/Charlie839242/SAS):
- [x] ScanNet
- [ ] Matterport3D
- [ ] nuScenes

Download the full pre-processed data (or you can choose the specific folder to download):
```bash
git lfs install
git clone https://huggingface.co/datasets/Charlie839242/SAS
```

The structure of the pre-processed data (e.g., ScanNet) is as follows. "scannet_multiview_lseg" and "scannet_multiview_seem" store the 3D point features from LSeg and SEEM respectively. "scannet_vocabulary" contain the generated images and the constructed capabilities. "scannet_multiview_fuse" is the combination of "scannet_multiview_lseg" and "scannet_multiview_seem" with "scannet_vocabulary" as the guide.
```
data
  └── scannet
      ├── fused_feat
      │   └── scannet_multiview_fuse
      ├── point_feat
      │   ├── scannet_multiview_lseg
      │   └── scannet_multiview_seem
      └── vocabulary
          └── scannet_vocabulary
```

### Extract Point Features
You can also extract 3D point features, and obtain "scannet_multiview_lseg" and "scannet_multiview_seem" on your own. 

TODO

## Model Capability Construction
You can also synthesize images and obtain "scannet_vocabulary" on your own. 
```bash
cd MCC
```

### Download SAM, LSeg and SEEM checkpoint
Download from [LSeg Checkpoint](https://drive.google.com/file/d/1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7/view) and place it in lseg_util folder. Then download ADEChallengeData2016.zip from [link](https://ade20k.csail.mit.edu/), unzip it, and place it in lseg_util folder. Download the SEEM checkpoint from [link](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt) and place it in seem_util folder. Download the SAM checkpoint from [link](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place it in sam_util folder.

### Generate synthesized images
```bash
python Stable_Diffusion/generate_any_class.py    # This will generate images in synthesized_img folder
```


### Compute the category embedding
You can skip this step and directly use the provided vocabualry_embedding.py.
```bash
python lseg_util/generate_text_embedding.py    # This will generate "vocabualry_embedding.py"
```


### Compute masks from LSeg
```bash
conda activate lseg
python lseg_util/lseg_infer.py    # This will generate masks in lseg_mask folder
```

### Compute masks from SEEM
```bash
conda activate seem
python seem_util/seem_infer.py    # This will generate masks in seem_mask folder
```

### Compute pseudo masks from SAM
```bash
python sam_util/generate_mask.py    # This will generate masks in refined_mask folder
```

### Compute mIOU
```bash
python miou/cal_miou.py --split=lseg    # This will generate miou in out folder and capability folder
python miou/cal_miou.py --split=seem    
```



## Feature Fusion
Once we obtain the 3D point features from LSeg and SEEM, as well as the model capabilities, we can start to fusion the point features from different datasets.

TODO

## Superpoint Extraction
TODO



## Train
TODO



## TODO List
- [x] Installation
- [x] Pre-processed data
- [x] Model capability construction
- [ ] The first stage of training
- [ ] The second stage of training
- [ ] Extraction of superpoints
- [ ] Code for extraction of point features from LSeg and SEEM
- [ ] Code and data for MatterPort3D
- [ ] Code and data for nuScenes


## Results


### 1. Evaluations on zero-shot 3D semantic segmentation

We compare SAS with both zero-shot and fully-supervised approaches on nuScenes, ScanNet and MatterPort3D using mIoU as metrics. Best results under each setting are shown bold. 
|Model | nuScenes | ScanNet | MatterPort3D |
|  :----  | :---: |  :---: | :---: |
|OpenScene | 42.1 | 54.2 | 43.4|
|GGSD| 46.1 | 56.5|40.1|
|Diff2Scene|- |  48.6 | 45.5|
|Seal| 45.0| -|-|
|OV3D| 44.6 | 57.3|45.8|
|**SAS**|**47.5** | **61.9**|**48.6**|

### 2. Gaussian segmentation

2D semantic segmentation results of 3D gaussian splatting on 12 scenes from the ScanNet v2 validation set. Comparison focuses on NeRF/3DGS-based methods using mIoU and mAcc.
|Model | Backbone | mIoU | mAcc | 
|  :----  | :---: | :---: | :---: |
|OpenSeg | EfficientNet | 53.4 | 75.1|
|LSeg | ViT | 56.1 |  74.5|
|LERF | NeRF+CLIP | 31.2 | 61.7|
|PVLFF | NeRF+LSeg | 52.9 | 67.0|
|LangSplat | 3DGS| 24.7 | 42.0|
|Feature3DGS | 3DGS+LSeg | 59.2 | 75.1|
|Semantic Gaussians | 3DGS+LSeg | 60.7 | 76.3|
|**SAS** | 3DGS+LSeg+SEEM | **63.9**| **79.9**|



### 3. 3D Instance segmentation

3D open-vocabulary instance segmentation results on ScanNet v2 validation set.
|Model | Semantic | mAP | AP@50 | AP@25 |
|  :----  | :---: | :---: | :---: |:---: |
|PointClip | Clip | - |4.5| 14.4|
|OpenIns3D | Yoloworld | 19.9 |28.9| **38.9**|
|OpenScene (2D/3D Ens.) | LSeg | 19.7 |25.9| 30.4|
|OpenScene (3D Distill) | LSeg | 19.8 |25.7|30.4|
|**SAS** | LSeg+SEEM | **25.1** |**31.7**|36.7|

## Citation
```
@article{li2025sas,
  title={SAS: Segment Any 3D Scene with Integrated 2D Priors},
  author={Li, Zhuoyuan and Lu, Jiahao and Deng, Jiacheng and Chang, Hanzhi and Wu, Lifan and Liang, Yanzhe and Zhang, Tianzhu},
  journal={arXiv preprint arXiv:2503.08512},
  year={2025}
}
```
