import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
del sys.path[5]
# print(sys.path)

import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange

from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
from encoding.models.sseg import BaseNet
import torchvision.transforms as transforms

import os
import torch
import glob
import math
import numpy as np
from PIL import Image
from torchvision import transforms


seed = 1457
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

#!### Dataset specific parameters #####
img_dim = (320, 240)
depth_scale = 1000.0
fx = 577.870605
fy = 577.870605
mx=319.5
my=239.5
#######################################
visibility_threshold = 0.25 # threshold for the visibility check


##############################
##### load the LSeg model ####

module = LSegModule.load_from_checkpoint(
    checkpoint_path='lseg_util/demo_e200.ckpt',
    data_path='lseg_util/',
    dataset='ade20k',
    backbone='clip_vitl16_384',
    aux=False,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0,
    batch_size=1,
    max_epochs=0,
    ignore_index=255,
    dropout=0.0,
    scale_inv=False,
    augment=False,
    no_batchnorm=False,
    widehead=True,
    widehead_hr=False,
    map_locatin="cpu",
    arch_option=0,
    block_depth=0,
    activation='lrelu',
)

# model
if isinstance(module.net, BaseNet):
    model = module.net
else:
    model = module

model = model.eval()
model = model.cpu()

model.mean = [0.5, 0.5, 0.5]
model.std = [0.5, 0.5, 0.5]

#############################################
# THE trick for getting proper LSeg feature for ScanNet
scales = ([1])
model.crop_size = 640
model.base_size = 640


evaluator = LSeg_MultiEvalModule(
    model, scales=scales, flip=True
    
).cuda() # LSeg model has to be in GPU
evaluator.eval()

transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )


import matplotlib.pyplot as plt
from tqdm import tqdm

CLASS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'other', 'ceiling']

text_feat = torch.load('lseg_util/vocabulary_embedding.pth').cuda().float()

for clas_ in CLASS:
    if clas_ == 'other':
        continue
    print(clas_)
    clas = CLASS.index(clas_)

    base_dir = 'synthesized_img/' + clas_.replace(' ', '_')
    img_path = os.path.join(base_dir, 'train_image')
    out_path = os.path.join(base_dir, 'lseg_mask')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for img in tqdm(os.listdir(img_path)):
        image = Image.open(os.path.join(img_path, img))
        image = np.array(image)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = evaluator.parallel_forward(image, '')
            feat_2d = outputs[0][0]
            feat_2d = feat_2d.permute(1, 2, 0)

        result = torch.einsum('hwc,nc->hwn', feat_2d, text_feat)
        result = result.argmax(dim=-1)

        mask_ = result == clas

        result[mask_] = 1
        result[~mask_] = 0

        result = result.cpu().numpy().astype(np.uint8)
        result = result * 255

        result = Image.fromarray(result)
        result = result.resize((512, 512), Image.NEAREST)
        result.save(os.path.join(out_path, img))