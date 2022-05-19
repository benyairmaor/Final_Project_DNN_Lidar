import pandas as pd
import copy
import open3d as o3d
import numpy as np
import torch
import Utilities as F
# Method to get all the problems from global file
def get_data_global(directory):
    
    headers = ['id', 'source', 'target', 'overlap', 't1', 't2','t3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
    read_file = pd.read_csv('Datasets/eth/' + directory + '_global.txt', sep=" ", header=0, names=headers)
    read_file.to_csv('Datasets/eth/' + directory + '_global.csv', sep=',')
    read_file = pd.DataFrame(read_file, columns=headers)        
    return read_file


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def draw_registration_result(source, target, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], width=1280, height=720)


def prepare_item(source_path, target_path, trans_init):
    
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path)) 
    # draw_registration_result(source, target, np.identity(4))
    # source.transform(trans_init)      
    # draw_registration_result(source, target, np.identity(4))
    return source, target


def preprocess_point_cloud(source, target, voxel_size):
    
    radius_normal = 5*voxel_size
    radius_feature = 10*voxel_size
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    # draw_registration_result(source_down, target_down, np.identity(4))
    
    # # Calculate normals
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
    
    # Calculate FPFHs
    pcd_fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=300))
    pcd_fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=300))
    
    return np.asarray(pcd_fpfh_source.data).T, np.asarray(pcd_fpfh_target.data).T


def get_data(idx, pcd_list):
    
    M = np.zeros((4, 4))
    for i in range(1, 13):
        idx_row = int((i - 1) / 4)
        idx_col = (i - 1) % 4
        M[idx_row, idx_col] = pcd_list.at[idx, 't' + str(i)]
    # o3d.geometry.TriangleMesh.create_coordinate_frame().get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    M[3, :] = [0, 0, 0, 1]  
    return pcd_list.at[idx, 'source'], pcd_list.at[idx, 'target'], pcd_list.at[idx, 'overlap'], M






path_to_dir = "apartment"
dir_name = path_to_dir

pcd_list = get_data_global(path_to_dir)
source, _, overlap, M = get_data(18, pcd_list)
source_path = 'Datasets/eth/' + dir_name + '/' + source
target_path = 'Datasets/eth/' + dir_name + '/' + source
source_, target_ = prepare_item(source_path, target_path, M)
source_arr, target_arr = np.asarray(source_.points), np.asarray(target_.points)
source_fpfh, target_fpfh = preprocess_point_cloud(source_, target_, 1)
source_fpfh_ = np.zeros((source_fpfh.shape[0], source_fpfh.shape[1]))
target_fpfh_ = np.zeros((target_fpfh.shape[0], target_fpfh.shape[1]))
source_fpfh = source_fpfh.T
target_fpfh = target_fpfh.T
source_fpfh_ = source_fpfh_.T
target_fpfh_ = target_fpfh_.T
# print(target_fpfh_.shape)
for i in range(source_fpfh_.shape[0]):
    for j in range(source_fpfh_.shape[1]):
        source_fpfh_[i][j] = np.exp(source_fpfh[i][j])/sum(np.exp(source_fpfh[i]))
        target_fpfh_[i][j] = np.exp(target_fpfh[i][j])/sum(np.exp(target_fpfh[i]))
    # source_fpfh_[i] = (source_fpfh[i] - np.min(source_fpfh[i])) / (np.max(source_fpfh[i]) - np.min(source_fpfh[i]) + 1e-12)
    # target_fpfh_[i] = (target_fpfh[i] - np.min(target_fpfh[i])) / (np.max(target_fpfh[i]) - np.min(target_fpfh[i]) + 1e-12)
    # print(np.sum(source_fpfh_[i], axis=0))
    # print(np.sum(target_fpfh_[i], axis=0))

source_fpfh_ = source_fpfh_.T
target_fpfh_ = target_fpfh_.T
source_fpfh = source_fpfh.T
target_fpfh = target_fpfh.T

# counter = 0
# counter222 =0
for s in source_fpfh_:
    for t in target_fpfh_:
        counter = 0
        for i in range(len(s)):
            if s[i] == t[i]:
                counter += 1
            else:
#                 # print("source", s)
#                 # print("target", t)
                s1 = s.reshape(1,source_fpfh.shape[1])
                t1 = t.reshape(1,target_fpfh.shape[1])
                s_ = torch.tensor(s1)
                t_ = torch.tensor(t1)
#                 print("einsum", torch.einsum('dn,dm->nm',s_.T, t_.T))
                print("dist", torch.dist(s_, t_))
#                 if  torch.einsum('dn,dm->nm',s_.T, t_.T) == 1:
#                     counter222 += 1
#                 break
#         if counter >= len(s) - 1:
#             print("match")
#             break
#     if counter >= len(s) - 1:
#         continue
#     else:
#         print("unmatch")
# print("x")
# print(counter222)

# scores = torch.einsum('dn,dm->nm',source_fpfh_.T, target_fpfh_.T)
# scores = scores.reshape(1,scores.shape[0], scores.shape[1])

for s in source_fpfh_:
    for t in target_fpfh_:
scores = torch.cdist(torch.tensor(source_fpfh_), torch.tensor(target_fpfh_))


# scores = np.zeros((1,source_fpfh.shape[0], target_fpfh.shape[0]))
scores = 1 - scores
dustBin = torch.tensor(1.)
num_iter = 1000  
scores = F.log_optimal_transport(scores=scores, alpha=dustBin, iters=num_iter)


# Get the matches with score above "match_threshold".
max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
indices0, indices1 = max0.indices, max1.indices
mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
zero = scores.new_tensor(0)
mscores0 = torch.where(mutual0, max0.values.exp(), zero)
mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
valid0 = mutual0 
valid1 = mutual1 & valid0.gather(1, indices1)
indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))