from PIL import Image
import numpy as np
import os 
from tqdm import tqdm
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='0'

def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='lseg or seem')
    parser.add_argument('--split', type=str, default='lseg', help='split: "lseg"| "seem"')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.split not in ['lseg', 'seem']:
        raise NotImplementedError


    for j in tqdm(os.listdir('synthesized_img')):
        base_path = os.path.join('synthesized_img', j)
        lseg_path = os.path.join(base_path, args.split + '_mask')
        gt_path = os.path.join(base_path, 'refined_mask')
        out_path = os.path.join(base_path, 'out')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        intersection_list = []
        union_list = []


        for i in os.listdir(lseg_path):
            lseg_mask = (np.array(Image.open(os.path.join(lseg_path, i))) / 255).astype(np.uint8)
            gt_mask = (np.array(Image.open(os.path.join(gt_path, i).replace('.jpg', '.png'))) / 255).astype(np.uint8)

            intersection = np.logical_and(gt_mask, lseg_mask).sum()
            union = np.logical_or(gt_mask, lseg_mask).sum()

            intersection_list.append(intersection)
            union_list.append(union)

        intersection = np.array(intersection_list).sum()
        union = np.array(union_list).sum()

        miou = intersection / union
        np.save(os.path.join(out_path, args.split + '_miou.npy'), miou)

    
    CLASS_NAME = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
                        'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                        'shower curtain', 'toilet', 'sink', 'bathtub', 'other', 'ceiling']


    miou_list = []

    for i in CLASS_NAME:
        if i == 'other':
            miou_list.append(0)
            continue
        path = os.path.join('synthesized_img', i.replace(' ', '_'), 'out', args.split + '_miou.npy')
        miou = np.load(path)
        miou_list.append(miou)

    miou = np.array(miou_list)
    np.save(os.path.join('capability',  args.split + '_miou.npy'), miou)