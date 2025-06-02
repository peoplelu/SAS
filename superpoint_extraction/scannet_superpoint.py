import os
import open3d as o3d
import torch
import numpy as np
import segmentator

def cal_sp(x):
    mesh = o3d.io.read_triangle_mesh(x)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces, kThresh=0.01, segMinVerts=60).numpy()
    return superpoint


path = 'superpoint_extraction/scannet_v2/scans'  # Your path to the raw scannet_v2 dataset
save_path = 'superpoint_extraction/scannet_superpoint'  # save path
for scene in os.listdir(path):
    ply_file_path = os.path.join(path, scene, scene+'_vh_clean_2.ply')
    name = ply_file_path.split('/')[-1].replace('.ply', '.pth')
    out_path = os.path.join(save_path, name)
    superpoint = cal_sp(ply_file_path)

    torch.save(superpoint, out_path)






