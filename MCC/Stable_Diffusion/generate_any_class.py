from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import json
import argparse
import multiprocessing as mp
import threading
from random import choice
import os
import argparse
from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
from tqdm import tqdm

LOW_RESOURCE = True 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3,tokenizer=None,device=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]],tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)

    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer



from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    return out.cpu()


def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.2 + mask_3d_color[mask] * 0.8
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb

def get_findContours(mask):
    mask_instance = (mask>0.5 * 1).astype(np.uint8) 
    ontours, hierarchy = cv2.findContours(mask_instance.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 0
    polygon_ins = []
    x,y,w,h = 0,0,0,0
    
    image_h, image_w = mask.shape[0:2]
    gt_kernel = np.zeros((image_h,image_w), dtype='uint8')
    for cnt in ontours:
        x_ins_t, y_ins_t, w_ins_t, h_ins_t = cv2.boundingRect(cnt)

        if w_ins_t*h_ins_t<250:
            continue
        cv2.fillPoly(gt_kernel, [cnt], 1)

    return gt_kernel
                        
def save_cross_attention(orignial_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None):
    
#     ("up", "down")
#     ("mid", "up", "down")

    orignial_image = orignial_image.copy()
    show = True
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode

    check_list = [decoder(int(i)) for i in tokenizer.encode(class_one)][1:-1]

    
    # "up", "down"
    attention_maps_8s = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select,prompts=prompts)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]
    
    
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]
    

    cam_dict = {}

    gt_kernel_final = np.zeros((512,512), dtype='float32')
    number_gt = 0
    for i in range(len(tokens)):
        class_current = decoder(int(tokens[i])) 
        
        if class_current not in check_list:
            continue
        
        image_8 = attention_maps_8s[:, :, i]
        image_8 = cv2.resize(image_8.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_8 = image_8 / image_8.max()
        
        image_16 = attention_maps[:, :, i]
        image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_16 = image_16 / image_16.max()
        
        image_32 = attention_maps_32[:, :, i]
        image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_32 = image_32 / image_32.max()
        
        image_64 = attention_maps_64[:, :, i]
        image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_64 = image_64 / image_64.max()
        
        if class_one == "sofa" or class_one == "train" or class_one == "tvmonitor":
            image = image_8
        elif class_one == "diningtable":
            image = image_16
        else:
            image = (image_16 + image_32 + image_64) / 3
        
        gt_kernel_final += image.copy()
        number_gt += 1

    if number_gt!=0:
        gt_kernel_final = gt_kernel_final/number_gt
        
        image = gt_kernel_final
        image = image - image.min()
        image = image / image.max()
        image[image>0.7]=1
        image[image<0.7]=0
        image = 255 * (image.astype(np.uint8))
        image = Image.fromarray(image)
        image.save(out_put+'.png')
        
    

    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
    
    
def run(prompts, controller, latent=None, generator=None,out_put = "",ldm_stable=None):

    images_here, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=generator, low_resource=LOW_RESOURCE)

    ptp_utils.view_images(images_here,out_put = out_put)
    return images_here, x_t



torch.cuda.set_device(2)
LOW_RESOURCE = True 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

tokenizer = ldm_stable.tokenizer



def sub_processor(class_name):
    num_image = 50
    classes = class_name
    image_cnt = 0

    if os.path.exists(os.path.join('synthesized_img', classes.replace(' ', '_'))):
        return None

    image_path = os.path.join('synthesized_img', classes.replace(' ', '_'), "train_image")
    npy_path = os.path.join('synthesized_img', classes.replace(' ', '_'), "npy")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    for rand in range(num_image):
        g_cpu = torch.Generator().manual_seed(image_cnt)

        prompts = ['a good photo of a ' + classes + ' in a room']
        # prompts = [classes]

        controller = AttentionStore()
        image_cnt+=1
        image, x_t = run(prompts, controller, latent=None,  generator=g_cpu,out_put = os.path.join(image_path,"image_{}_{}.jpg".format(classes.replace(' ', '_'),image_cnt)),ldm_stable=ldm_stable)
        save_cross_attention(image[0].copy(),controller, res=32, from_where=("up", "down"),out_put = os.path.join(npy_path,"image_{}_{}".format(classes.replace(' ', '_'),image_cnt)),image_cnt=image_cnt,class_one=classes,prompts=prompts,tokenizer=tokenizer)
        # torch.cuda.empty_cache()

CLASS_NAME = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
                        'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                        'shower curtain', 'toilet', 'sink', 'bathtub', 'ceiling']


for class_name in tqdm(CLASS_NAME):
    sub_processor(class_name)



