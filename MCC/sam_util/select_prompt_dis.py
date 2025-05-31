# 导入必要的库
import numpy as np
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image


# 定义一个函数，用于计算离边缘最远的点
def farthest_point(mask):
    # 使用距离变换函数，计算每个像素到最近背景点（即0）的距离
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    #可视化距离变换图
    plt.imshow(dist)#, cmap='hot')
    # plt.colorbar()
    plt.savefig('heatmap.png') 
    # 找出距离最大的像素，即离边缘最远的点
    max_dist = np.max(dist)
    max_idx = np.where(dist == max_dist)
    plt.close()
    # 返回该点的坐标
    return max_idx[0][0], max_idx[1][0]

def select_prompt_points_for_mask(mask):
    # mask: [512*512]
    mask = np.array(mask)
    # 使用连通域分析函数，获取连通域数量，标签和统计信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # 根据连通域数量的不同，分别处理三种情况
    if num_labels == 2: # 只有一个连通域（除了背景）
        # 在该区域中依次找出三个离边缘最远的点A, B, C
        A = farthest_point(mask)
        # 将A点置为边缘（即0）
        mask[A[0], A[1]] = 0
        B = farthest_point(mask)
        # 将B点置为边缘（即0）
        mask[B[0], B[1]] = 0
        C = farthest_point(mask)
        # 输出A, B, C点的坐标
        points = [[p[1],p[0]] for p in [A,B,C]]
        # print(points)
        return points
    elif num_labels == 3: # 有两个连通域（除了背景）
        # 分别在最大和第二大的区域中找出两个和一个离边缘最远的点
        # 获取每个区域的面积，并按降序排序
        areas = stats[:, cv2.CC_STAT_AREA]
        sorted_areas = np.sort(areas)[::-1]
        # 获取最大和第二大区域的标签
        label1 = np.where(areas == sorted_areas[1])[0][0]
        label2 = np.where(areas == sorted_areas[2])[0][0]
        # 在最大区域中找出两个离边缘最远的点A, B
        mask1 = np.where(labels == label1, 255, 0).astype(np.uint8)
        A = farthest_point(mask1)
        # 将A点置为边缘（即0）
        mask1[A[0], A[1]] = 0
        B = farthest_point(mask1)
        # 在第二大区域中找出一个离边缘最远的点C
        mask2 = np.where(labels == label2, 255, 0).astype(np.uint8)
        C = farthest_point(mask2)
        # 输出A, B, C点的坐标
        points = [[p[1],p[0]] for p in [A,B,C]]

        # print(points)
        return points
    else: # 有三个或以上的连通域（除了背景）
        # 在最大的三个区域中各找出一个离边缘最远的点
        # 获取每个区域的面积，并按降序排序
        areas = stats[:, cv2.CC_STAT_AREA]
        sorted_areas = np.sort(areas)[::-1]
        # 获取最大的三个区域的标签
        label1 = np.where(areas == sorted_areas[1])[0][0]
        label2 = np.where(areas == sorted_areas[2])[0][0]
        label3 = np.where(areas == sorted_areas[3])[0][0]
        # 在每个区域中找出一个离边缘最远的点A, B, C
        mask1 = np.where(labels == label1, 255, 0).astype(np.uint8)
        A = farthest_point(mask1)
        mask2 = np.where(labels == label2, 255, 0).astype(np.uint8)
        B = farthest_point(mask2)
        mask3 = np.where(labels == label3, 255, 0).astype(np.uint8)
        C = farthest_point(mask3)
        # 输出A, B, C点的坐标
        # print("A:", A)
        # print("B:", B)
        # print("C:", C)
        points = [[p[1],p[0]] for p in [A,B,C]]

        return points

