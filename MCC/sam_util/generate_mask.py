from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import os 
os.environ['CUDA_VISIBLE_DEVICES']='2'
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from select_prompt_dis import select_prompt_points_for_mask



sam = sam_model_registry["vit_h"](checkpoint="sam_util/sam_vit_h_4b8939.pth").cuda()
predictor = SamPredictor(sam) 



for i in tqdm(os.listdir('synthesized_img')):
    path = os.path.join('synthesized_img', i)


    mask_path = os.path.join(path,'npy') # diffu mask的路径
    image_path = os.path.join(path, 'train_image')
    save_path = os.path.join(path, 'refined_mask') # 存储refined mask
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if 1:
        for mask_name in tqdm(os.listdir(mask_path)): 
            image_name = mask_name.replace('.png','.jpg')
            init_mask = Image.open(os.path.join(mask_path, mask_name))
            image = cv2.imread(os.path.join(image_path, image_name))
            
            ## 预防边角出现mask的情况，有的情况四个角会出现mask
            init_mask = np.array(init_mask)
            init_mask[:70,:] = 0
            init_mask[-70:,:] = 0
            init_mask[:,-70:] = 0
            init_mask[:,:70] = 0
            if init_mask.sum()<255*10:
                init_mask[240:242,240:242]=255
                init_mask[240:242,270:272]=255
                init_mask[270:272,240:242]=255
                init_mask[270:272,270:272]=255
            init_mask = Image.fromarray(init_mask)
            
            # 使用距离变换挑选prompt points，使用3个点
            input_points = select_prompt_points_for_mask(init_mask) 
            input_points = np.array(input_points)
            input_label = np.array([1,1,1])
            
            #SAM分割，仅产生一个mask
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_label,
                multimask_output=False,
            )
            mask = 255 * ((masks[0]).astype(np.uint8))
            
            # 在mask中用红色点高亮这3个点,进行可视化观察
            # plt.imshow(mask, cmap="gray")
            # plt.scatter([p[0] for p in input_points], [p[1] for p in input_points], c="b", s=10)
            # plt.savefig(os.path.join(save_path, 'ori_cood'+mask_name))
            # plt.close()
            
            ## 保存生成的mask
            mask = Image.fromarray(mask)
            mask.save(os.path.join(save_path, mask_name))
            mask.close()
            plt.close()