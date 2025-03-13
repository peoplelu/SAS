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

## :rocket: News

**11/Mar/2025**: We release our paper to [Arxiv](https://arxiv.org/abs/2503.08512).


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
