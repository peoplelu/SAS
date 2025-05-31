import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

# import sys
# sys.path.append('/ssd/lizhuoyuan/temp/temp/Segment-Everything-Everywhere-All-At-Once')
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import torch
from torchvision import transforms
import argparse
import whisper
import numpy as np
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from gradio import processing_utils
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES, supple


import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
import os
import glob
import subprocess
from PIL import Image
import random


opt = load_opt_from_config_files(["seem_util/configs/seem/focall_unicl_lang_demo.yaml"])
opt = init_distributed(opt)

pretrained_pth = "seem_util/seem_focall_v0.pt"

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + supple, is_eval=True)




@torch.no_grad()
def inference(image, model):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return interactive_infer_image(model, image)

t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
print(metadata)
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]


def interactive_infer_image(model, image):
    image_ori = transform(image)
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()



    data = {"image": images, "height": height, "width": width}
    tasks = ["Panoptic"]
    
    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False


    batch_inputs = [data]
    if 'Panoptic' in tasks:
        model.model.metadata = metadata
        results, mask_emb = model.model.evaluate(batch_inputs)
        mask_emb = mask_emb.squeeze()
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        real_id = torch.tensor(results[-1]['panoptic_seg'][2])
        keep = results[-1]['panoptic_seg'][3]


        sem_seg = results[-1]['sem_seg']


        return None, pano_seg, pano_seg_info, mask_emb, keep, real_id, sem_seg


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

CLASS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'other', 'ceiling']

scannet_emb = torch.load('lseg_util/vocabulary_embedding.pth').float().cuda()
t_emb = torch.load('seem_util/seem_text_embedding.pth').float().cuda()

for clas_ in CLASS:
    if clas_ == 'other':
        continue
    print(clas_)
    clas = CLASS.index(clas_)

    base_dir = 'synthesized_img/' + clas_.replace(' ', '_')
    img_path = os.path.join(base_dir, 'train_image')
    out_path = os.path.join(base_dir, 'seem_mask')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for img in tqdm(os.listdir(img_path)):
        image = Image.open(os.path.join(img_path, img)).convert('RGB')
        x, pano_seg, pano_seg_info, mask_emb, keep, real_id, sem_seg = inference(image, model)

        sem_seg = sem_seg.argmax(dim=0)

        sem_seg = t_emb[sem_seg, :].float()



        sem_seg = sem_seg @ scannet_emb.t()
        # print(sem_seg.shape)
        sem_seg = sem_seg.argmax(dim=-1)

        mask_ = sem_seg == clas

        sem_seg[mask_] = 1
        sem_seg[~mask_] = 0

        result = sem_seg.cpu().numpy().astype(np.uint8)
        result = result * 255

        result = Image.fromarray(result)
        result = result.resize((512, 512), Image.NEAREST)
        result.save(os.path.join(out_path, img))



