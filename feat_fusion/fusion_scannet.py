import torch
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='2'
from tqdm import tqdm


lseg_path = 'data/scannet/point_feat/scannet_multiview_lseg'
seem_path = 'data/scannet/point_feat/scannet_multiview_seem'

text_features = torch.load('MCC/lseg_util/vocabulary_embedding.pth').cuda()
lseg_miou = torch.from_numpy(np.load('data/scannet/vocabulary/scannet_vocabulary/capability/lseg_miou.npy')).cuda().float()
seem_miou = torch.from_numpy(np.load('data/scannet/vocabulary/scannet_vocabulary/capability/seem_miou.npy')).cuda().float()


for scene_name in tqdm(os.listdir(lseg_path)):
    lseg_scene = os.path.join(lseg_path, scene_name)
    seem_scene = os.path.join(seem_path, scene_name)

    lseg_data = torch.load(lseg_scene)
    seem_data = torch.load(seem_scene)

    lseg_feat = lseg_data['feat'].cuda()
    seem_feat = seem_data['feat'].cuda()

    lseg_logit = (lseg_feat.half() @ text_features.t()).argmax(dim=-1)
    seem_logit = (seem_feat.half() @ text_features.t()).argmax(dim=-1)

    mask_1 = lseg_logit == 19
    mask_1 = ~mask_1

    lseg_weight = lseg_miou[lseg_logit[mask_1]] + lseg_miou[seem_logit[mask_1]]
    seem_weight = seem_miou[seem_logit[mask_1]] + seem_miou[lseg_logit[mask_1]]
    weight_sum = lseg_weight + seem_weight


    lseg_weight, seem_weight = lseg_weight / weight_sum, seem_weight / weight_sum
    
    weight = torch.concat([lseg_weight[:, None], seem_weight[:, None]], dim=-1)
    weight = torch.nn.functional.softmax(weight/0.05, dim=-1)

    lseg_weight, seem_weight = weight[:, 0], weight[:, 1]


    new_feat = lseg_feat[mask_1] * lseg_weight[:, None] + seem_feat[mask_1] * seem_weight[:, None]

    lseg_feat[mask_1] = new_feat.half()
    lseg_feat = lseg_feat.cpu().detach()

    lseg_data['feat'] = lseg_feat

    torch.save(lseg_data, os.path.join('data/scannet/target_path', scene_name))


