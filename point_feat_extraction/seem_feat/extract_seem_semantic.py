import os
os.environ['CUDA_VISIBLE_DEVICES']='4'

# import sys
# sys.path.append('/ssd/lizhuoyuan/Segment-Everything-Everywhere-All-At-Once')
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
import matplotlib.pyplot as plt


opt = load_opt_from_config_files(["configs/seem/focall_unicl_lang_demo.yaml"])
opt = init_distributed(opt)

pretrained_pth = "seem_focall_v0.pt"

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + supple, is_eval=True)


colors = np.random.randint(0, 256, (len(COCO_PANOPTIC_CLASSES+supple), 3))

n = []
n.append(transforms.Resize((240, 320), interpolation=Image.NEAREST))
transform_n = transforms.Compose(n)



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






for scene in os.listdir(path):
    if scene == 'intrinsics.txt':
        continue
    scene_path = os.path.join(path, scene)+'/color'
    for img in os.listdir(scene_path):
        img_path = os.path.join(scene_path, img)
        img_name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '')
        mask_path = img_path[:23] + '/sem_seg/' + img_name + '.pth'
        mask_img_path = img_path[:23] + '/sem_img/' + img_name + '.png'

        print(mask_path)
        if os.path.exists(mask_path) and os.path.exists(mask_img_path):
            print(mask_path+' already exist!')
            continue


        if not os.path.exists(img_path[:23] + '/sem_seg'):
            os.mkdir(img_path[:23] + '/sem_seg')  

        if not os.path.exists(img_path[:23] + '/sem_img'):
            os.mkdir(img_path[:23] + '/sem_img')  

        image = Image.open(img_path).convert('RGB')
        x, pano_seg, pano_seg_info, mask_emb, keep, real_id, sem_seg = inference(image, model)

        if pano_seg_info == []:     # no masks detected
            continue

        sem_seg = sem_seg.argmax(dim=0)
        sem_seg = transform_n(sem_seg[None, None, :, :]).squeeze().cpu()
        sem_seg = sem_seg.to(torch.uint8)


        sem_seg_img = colors[sem_seg.numpy()]

        plt.imshow(sem_seg_img.astype(np.uint8))


        torch.save(sem_seg, mask_path)
        plt.savefig(mask_img_path)
        plt.close()

