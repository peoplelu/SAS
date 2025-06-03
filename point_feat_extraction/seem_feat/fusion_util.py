import os
import torch
import glob
import math
import numpy as np
from PIL import Image
from torchvision import transforms
import clip
import nltk

COCO_PANOPTIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

supple = ['sofa', 'bookshelf', 'desk', 'shower curtain', 'bathtub', 'others']

cls_139 = COCO_PANOPTIC_CLASSES + supple
cls_139 = [i.replace('-other','').replace('-merged','').replace('-stuff','') for i in cls_139]


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic



def extract_seem_img_feature(img_dir, transform, evaluator, clip_pretrained, label=''):
    temp = img_dir.split('/')
    mask_dir = os.path.join(temp[0], temp[1], temp[2], 'seem_sem_seg', temp[4].split('.')[0]+'.pth')
    pano_mask_dir = os.path.join(temp[0], temp[1], temp[2], 'pano_seg', temp[4].split('.')[0]+'.pth')
    pano_caption_dir = os.path.join(temp[0], temp[1], temp[2], 'caption', temp[4].split('.')[0]+'.pth')

    if not (os.path.exists(mask_dir) and os.path.exists(pano_mask_dir)):
        return None

    # load masks
    mask = torch.load(mask_dir).squeeze().to(torch.int64)
    pano_mask = torch.load(pano_mask_dir).squeeze().to(torch.int64)
    captions = torch.load(pano_caption_dir)[0]['captions']

    # preprocess mask
    n = []
    n.append(transforms.Resize((240, 320), interpolation=Image.NEAREST))
    transform_n = transforms.Compose(n)
    pano_mask = transform_n(pano_mask[None, None, :, :]).squeeze()

    tokens = [nltk.word_tokenize(i) for i in captions]
    tags = [nltk.pos_tag(i) for i in tokens]
    adj = [[a for a, b in i if b == 'JJ'] for i in tags]
    adj = [' and '.join(i) for i in adj]
    adj = ['SsSs' if i == '' else i for i in adj]

    adv = []
    for i in tokens:
        try: 
            raise NotImplementedError
            # index = i.index('on')
            # if i[-1] == 'on':
            #     adv.append('AaAa')
            # else:
            #     adv.append(' '.join(i[index:]))
        except:
            adv.append('AaAa')

    original_subject = mask
    original_subject = original_subject.reshape((240 * 320)).tolist()
    original_subject = [cls_139[i] for i in original_subject]

    adj_map = pano_mask
    adj_map = adj_map.reshape((240 * 320)).tolist()
    adj_map = [adj[i] for i in adj_map]

    adv_map = pano_mask
    adv_map = adv_map.reshape((240 * 320)).tolist()
    adv_map = [adv[i] for i in adv_map]

    pixel_captions = ['a ' + i + ' ' + j + ' ' + k for i, j, k in zip(adj_map, original_subject, adv_map)]
    pixel_captions = [i.replace('SsSs ', '').replace(' AaAa', '') + ' in a scene' for i in pixel_captions]


    pixel_captions_set = list(set(pixel_captions))

    for i, v in enumerate(pixel_captions_set):
        for j in range(76800):
            if pixel_captions[j] == v:
                pixel_captions[j] = i

    pixel_captions = torch.tensor(pixel_captions).cuda().reshape(240, 320).to(torch.int64)

    with torch.no_grad():
        labels = []
        for line in pixel_captions_set:
            label = line
            labels.append(label)
        text = clip.tokenize(labels)
        text = text.cuda()
        text_features = clip_pretrained.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    feat_2d = text_features[pixel_captions, :]

    feat_2d = feat_2d.permute(2, 0, 1).half()

    return feat_2d


def save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire = mask_entire & mask

        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
        },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')


class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None: # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def obtain_intr_extr_matterport(scene):
    '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''

    img_dir = os.path.join(scene, 'color')
    pose_dir = os.path.join(scene, 'pose')
    intr_dir = os.path.join(scene, 'intrinsic')
    img_names = sorted(glob.glob(img_dir+'/*.jpg'))

    intrinsics = []
    extrinsics = []
    for img_name in img_names:
        name = img_name.split('/')[-1][:-4]

        extrinsics.append(np.loadtxt(os.path.join(pose_dir, name+'.txt')))
        intrinsics.append(np.loadtxt(os.path.join(intr_dir, name+'.txt')))

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics

def get_matterport_camera_data(data_path, locs_in, args):
    '''Get all camera view related infomation of Matterport3D.'''

    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]

    scene = os.path.join(args.data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (cam_loc[:, 0] > bbox_l[0]) & (cam_loc[:, 0] < bbox_h[0]) & \
                    (cam_loc[:, 1] > bbox_l[1]) & (cam_loc[:, 1] < bbox_h[1]) & \
                    (cam_loc[:, 2] > bbox_l[2]) & (cam_loc[:, 2] < bbox_h[2])

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    # some regions have no views inside, we consider it differently for test and train/val
    if args.split == 'test' and num_img == 0:
        print('no views inside {}, take the nearest 100 images to fuse'.format(scene_id))
        #! take the nearest 100 views for feature fusion of regions without inside views
        centroid = (bbox_l+bbox_h)/2
        dist_centroid = np.linalg.norm(cam_loc-centroid, axis=-1)
        ind_in_scene = np.argsort(dist_centroid)[:100]
        img_names_in = img_names[ind_in_scene]
        intrinsics_in = intrinsics[ind_in_scene]
        extrinsics_in = extrinsics[ind_in_scene]
        num_img = 100

    img_names_in = img_names_in.tolist()

    return intrinsics_in, extrinsics_in, img_names_in, scene_id, num_img