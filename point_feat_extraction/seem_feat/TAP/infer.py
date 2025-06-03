import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from tokenize_anything import engine
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
from tokenize_anything.models.easy_build import model_registry

import sys

class Predictor(object):
    """Predictor."""

    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = kwargs
        self.batch_size = kwargs.get("batch_size", 256)
        self.model.concept_projector.reset_weights(kwargs["concept_weights"])
        self.model.text_decoder.reset_cache(max_batch_size=self.batch_size)

    def preprocess_images(self, imgs):
        """Preprocess the inference images."""
        im_batch, im_shapes, im_scales = [], [], []
        for img in imgs:
            scaled_imgs, scales = im_rescale(img, scales=[1024])
            im_batch += scaled_imgs
            im_scales += scales
            im_shapes += [x.shape[:2] for x in scaled_imgs]
        im_batch = im_vstack(im_batch, self.model.pixel_mean_value, size=(1024, 1024))
        im_shapes = np.array(im_shapes)
        im_scales = np.array(im_scales).reshape((len(im_batch), -1))
        im_info = np.hstack([im_shapes, im_scales]).astype("float32")
        return im_batch, im_info

    @torch.inference_mode()
    def get_results(self, examples):
        """Return the results."""
        # Preprocess images and prompts.
        imgs = [example["img"] for example in examples]
        points = np.concatenate([example["points"] for example in examples])
        im_batch, im_info = self.preprocess_images(imgs)
        num_prompts = points.shape[0] if len(points.shape) > 2 else 1
        batch_shape = im_batch.shape[0], num_prompts // im_batch.shape[0]
        batch_points = points.reshape(batch_shape + (-1, 3))
        batch_points[:, :, :, :2] *= im_info[:, None, None, 2:4]
        batch_points = batch_points.reshape(points.shape)
        # Predict tokens and masks.
        inputs = self.model.get_inputs({"img": im_batch})
        inputs.update(self.model.get_features(inputs))
        outputs = self.model.get_outputs(dict(**inputs, **{"points": batch_points}))
        # Select final mask.
        iou_pred = outputs["iou_pred"].cpu().numpy()
        point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
        rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
        mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
        iou_scores = outputs["iou_pred"][mask_index].cpu().numpy().reshape(batch_shape)
        # Upscale masks to the original image resolution.
        mask_pred = outputs["mask_pred"][mask_index].unsqueeze_(1)
        mask_pred = self.model.upscale_masks(mask_pred, im_batch.shape[1:-1])
        mask_pred = mask_pred.view(batch_shape + mask_pred.shape[2:])
        # Predict concepts.
        concepts, scores = self.model.predict_concept(outputs["sem_embeds"][mask_index])
        concepts, scores = [x.reshape(batch_shape) for x in (concepts, scores)]
        # Generate captions.
        sem_tokens = outputs["sem_tokens"][mask_index]
        captions = self.model.generate_text(sem_tokens).reshape(batch_shape)
        # Postprocess results.
        results = []
        for i in range(batch_shape[0]):
            pred_h, pred_w = im_info[i, :2].astype("int")
            masks = mask_pred[i : i + 1, :, :pred_h, :pred_w]
            masks = self.model.upscale_masks(masks, imgs[i].shape[:2]).flatten(0, 1)
            results.append(
                {
                    "scores": np.stack([iou_scores[i], scores[i]], axis=-1),
                    "masks": masks.gt(0).cpu().numpy().astype("uint8"),
                    "concepts": concepts[i],
                    "captions": captions[i],
                }
            )
        return results

model_type = 'tap_vit_h'
checkpoint = 'TAP/models/tap_vit_h_v1_1.pkl'
concept = 'TAP/concepts/merged_2560.pkl'
device = [0]
port = 2030


kwargs={
    "model_type": model_type,
    "weights": checkpoint,
    "concept_weights": concept,
    "device": device[0],
    "predictor_type": Predictor,
    "verbose": 1,
}

builder = model_registry[model_type]
model = builder(device=device[0], checkpoint=checkpoint)

predicter = Predictor(model, kwargs)

def farthest_point(mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    max_dist = np.max(dist)
    max_idx = np.where(dist == max_dist)
    plt.close()

    return [max_idx[1][0], max_idx[0][0]]


path = 'scannet/scannet_2d'

for scene in os.listdir(path):
    if scene == 'intrinsics.txt':
        continue
    scene_path = os.path.join(path, scene)+'/color'
    for img in os.listdir(scene_path):
        img_path = os.path.join(scene_path, img)
        img_name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '')
        mask_path = img_path[:31] + '/pano_seg/' + img_name + '.pth'
        out_path = img_path[:31] + '/caption/' + img_name + '.pth'
        print(mask_path)
        if not os.path.exists(mask_path):
            print('no cooresponding pth!')
            continue

        
        if os.path.exists(out_path):
            print(out_path+' already exist!')
            continue

        if not os.path.exists(img_path[:31] + '/caption'):
            os.mkdir(img_path[:31] + '/caption')  


        points = []
        
        mask = torch.load(mask_path)
        # mask = mask['panoptic_seg'].cpu().numpy()
        mask = mask.cpu().numpy()
        max_cls = mask.max()
        for i in range(1, max_cls+1):
            mask_ = (mask==i).astype(np.uint8)
            point = farthest_point(mask_)
            point.append(1)
            point = np.array([point, [0, 0, 4]], dtype=np.float32)[None, :, :]
            points.append(point)
        try:
            points = np.concatenate(points, axis=0)
        except:
            continue

        img = cv2.imread(img_path)

        examples = [{}]
        examples[0]['img'] = img
        examples[0]['points'] = points

        results = predicter.get_results(examples)

        torch.save(results, out_path)

        