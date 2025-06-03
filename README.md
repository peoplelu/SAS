<div align="center">
 
# SAS: Segment Any 3D Scene with Integrated 2D Priors
![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://arxiv.org/abs/2503.08512'><img src='https://img.shields.io/badge/arXiv-2503.08512-b31b1b.svg'></a> &nbsp;
 <a href='https://peoplelu.github.io/SAS.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 [![Hugging Face Data](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/datasets/Charlie839242/SAS)&nbsp;

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

![teaser](assets/teaser_00.jpg)
</div>

## :rocket: News

**11/Mar/2025**: We release our paper to [Arxiv](https://arxiv.org/abs/2503.08512).



## 🚀 Quick Start
<details> <summary> 🛠️ Installation </summary>
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
pip install datasets, diffusers, timm, transformers, clip_interrogator
```
</details>

<details> <summary> 🔧 Dataset Preparation </summary>
 
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


#### LSeg features of ScanNet
This part of code is included in "point_feat_extraction/lseg_feat".

First, download LSeg weight [demo_e200.ckpt](https://github.com/isl-org/lang-seg) and put it in checkpoint folder. Then download ADEChallengeData2016.zip from [link](https://ade20k.csail.mit.edu/), unzip it, and place it in dataset folder.
Download the raw ScanNet 2D images from [OpenScene](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip) and ScanNet 3D data from [OpenScene](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip), and put them under scannet folder. 

```bash
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip
```

Then file strcuture is as follows:

```
lseg_feat
├── checkpoints
│   └── demo_e200.ckpt
├── dataset
│   └── ADEChallengeData2016
│   │   ├── ...
│   │   ├── ...
│   │   └── ...
├── scannet
│   ├── scannet_2d
│   │   ├── ...
│   │   ├── ...
│   │   └── ...
│   ├── scannet_3d
│   │   ├── ...
│   │   ├── ...
│   │   └── ...
```

Then execute the following command to extract per-point features of scannet  from LSeg:

```bash
cd point_feat_extraction/lseg_feat
conda activate lseg
python fusion_scannet.py
```

This will generate features from LSeg in "scannet_multiview_lseg" folder.




#### SEEM features of ScanNet
This part of code is included in "point_feat_extraction/seem_feat".

Download the SEEM checkpoint from [link](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt) and place it in seem_feat folder. Then, download the raw ScanNet 2D images from [OpenScene](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip) and ScanNet 3D data from [OpenScene](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip), and put them under scannet folder.

```bash
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip
```

Then file strcuture is as follows:

```
seem_feat
├── seem_focall_v0.pt
├── scannet
│   ├── scannet_2d
|   │   ├── scene0000_00
|   |   │   ├── color
|   |   │   ├── depth
|   |   │   ├── label
|   |   │   └── pose
|   │   ├── scene0000_01
|   |   │   ├── ...
|   |   │   └── ...
│   ├── scannet_3d
│   │   ├── ...
│   │   ├── ...
│   │   └── ...
```

First, execute the following command to extract the panoptic segmentation result of each 2D image from SEEM:

```bash
cd point_feat_extraction/seem_feat
conda activate seem
python extract_seem_pano.py
python extract_seem_semantic.py
```

Now, the file structure becomes:
```
seem_feat
├── scannet
│   ├── scannet_2d
|   │   ├── scene0000_00
|   |   │   ├── color
|   |   │   ├── depth
|   |   │   ├── label
|   |   │   ├── pose
|   |   │   ├── sem_seg
|   |   │   ├── sem_seg_img
|   |   │   ├── pano_seg
└── └── └── └── pano_seg_img
```

Second, execute the following code that utilizes [TAP](https://github.com/baaivision/tokenize-anything) to generate captions for masks from SEEM. Before this, download [TAP checkpoint](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_h_v1_1.pkl) and palce it in TAP/models/tap_vit_h_v1_1.pkl and 

```bash
conda create -n ta python=3.8
conda activate ta
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install packaging, ninja
pip install flash-attn --no-build-isolation
pip install git+ssh://git@github.com/baaivision/tokenize-anything.git

cd point_feat_extraction/seem_feat
python TAP/infer.py
```

Finally, execute the following code to encode the extracted captions of each mask:
```bash
cd point_feat_extraction/seem_feat
python fusion_scannet.py
```
This will generate features from SEEM in "scannet_multiview_seem" folder.



</details>



<details> <summary> 🎇 Model Capability Construction </summary>

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

</details>



<details> <summary> 🎆 Feature Fusion </summary>


To integrate the LSeg features and the SEEM features of the ScanNet dataset using the constructed capability as the guide, execute the following command:

```bash
python feat_fusion/fusion_scannet.py
```

</details>



<details> <summary> 🎥 Training </summary>

### Superpoint extraction

To extract superpoints of each scene in ScanNetv2 dataset, you should first download the raw [ScanNet v2](http://www.scan-net.org/) dataset to obtain the .ply file of each scene. The ScanNet v2 dataset structure is as follows:


```
superpoint_extraction
├── scannet_v2
│   ├── intrinsics.txt
│   ├── scene0000_00
│   │   ├── label-filt
│   │   ├── scene0000_00_2d-instance-filt.zip
│   │   ├── scene0000_00_2d-instance.zip
│   │   ├── scene0000_00_2d-label-filt.zip
│   │   ├── scene0000_00_2d-label.zip
│   │   ├── scene0000_00.aggregation.json
│   │   ├── scene0000_00.txt
│   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
│   │   ├── scene0000_00_vh_clean_2.labels.ply
│   │   ├── scene0000_00_vh_clean_2.ply
│   │   ├── scene0000_00_vh_clean.aggregation.json
│   │   ├── scene0000_00_vh_clean.ply
│   │   └── scene0000_00_vh_clean.segs.json
│   ├── scene0000_01
│   │   ├── ...
│   │   ├── ...
│   │   ├── ...
```

Then build the cpp lib for superpoint extraction:

```bash
cd csrc && mkdir build && cd build

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

make && make install # after install, please do not delete this folder (as we only create a symbolic link)
```

Then execute the following command to extract superpoints. The superpoint-related code is built upon [segmentator](https://github.com/Karbo123/segmentator).

```bash
python superpoint_extraction/scannet_superpoint.py
```

### Train
 TODO

</details>

<details> <summary> 🌟 Evaluation </summary>

 TODO
</details>







## TODO List
- [x] Installation
- [x] Pre-processed data
- [x] Model capability construction
- [ ] The first stage of training
- [ ] The second stage of training
- [ ] Code for evaluation
- [x] Extraction of superpoints
- [x] Code for extraction of point features from LSeg and SEEM
- [ ] Code and data for MatterPort3D
- [ ] Code and data for nuScenes





## 📜 Citation
```
@article{li2025sas,
  title={SAS: Segment Any 3D Scene with Integrated 2D Priors},
  author={Li, Zhuoyuan and Lu, Jiahao and Deng, Jiacheng and Chang, Hanzhi and Wu, Lifan and Liang, Yanzhe and Zhang, Tianzhu},
  journal={arXiv preprint arXiv:2503.08512},
  year={2025}
}
```


## 🤝 Acknowledgements
Our code is based on [OpenScene](https://github.com/pengsongyou/openscene). We thank the authors for their excellent work!
