import numpy as np
import open3d as o3d
import copy
import pandas as pd
import ot
import os
import torch
from dgl.geometry import farthest_point_sampler
import matplotlib.pyplot as plt

######################################################################
##################         General Functions        ##################
######################################################################

def savePCDS(source,target,title,pathToSave,trans,directory):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.05, 0.9, 0.05])
    target_temp.paint_uniform_color([0.9, 0.05, 0.05])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible = False,height=720,width=1280)
    source_temp.transform(trans)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("images/"+ directory +".json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    img = vis.capture_screen_float_buffer(True)
    if not os.path.exists(pathToSave):
        os.makedirs(pathToSave)
    plt.imsave(pathToSave+"/"+title+".png",np.asarray(img), dpi = 1)

# Method get the data from global file or POC file
def get_data_global(directory, POC):
    headers = ['id', 'source', 'target', 'overlap', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
    if POC:
        read_file = pd.read_csv('Datasets/eth/' + directory + '_global_POC.txt', sep=" ", header=0, names=headers)
        read_file.to_csv('Datasets/eth/' + directory + '_global_POC.csv', sep=',')
    else:
        read_file = pd.read_csv('Datasets/eth/' + directory + '_global.txt', sep=" ", header=0, names=headers)
        read_file.to_csv('Datasets/eth/' + directory + '_global.csv', sep=',')
    read_file = pd.DataFrame(read_file, columns=headers)
    
    M = np.zeros((len(read_file), 4, 4))
    for row in range(len(read_file)):
        for i in range(1, 13):
            idx_row = int((i - 1) / 4)
            idx_col = (i - 1) % 4
            M[row, idx_row, idx_col] = read_file['t' + str(i)][row]
        M[row, 3, :] = [0, 0, 0, 1]
    return read_file['source'], read_file['target'], read_file['overlap'], M
    

# For draw source & target point cloud.
def draw_registration_result(source, target, transformation, title):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], title, 1280, 720)


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud_voxel(pcd, voxel_size):
    radius_normal = voxel_size * 5
    radius_feature = voxel_size * 10
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
    return pcd_down, pcd_fpfh


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud_keypoint(pcd, voxel_size):
    radius_normal = voxel_size * 5
    radius_feature = voxel_size * 10
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
    return pcd_fpfh


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud_fartest_point(pcd,voxel_size,farthest_size):
    radius_normal = voxel_size * 5
    radius_feature = voxel_size * 10
    pcd_down = farthest_point(pcd, int(farthest_size * np.asarray(pcd.points).shape[0]))
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
    return pcd_down, pcd_fpfh


# For pre prossecing the point cloud - compute voxel.
def  preprocess_point_cloud_for_test(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


def findCorrZeroOne(source, target, distanceThreshold):
    # prepare list and copy data.
    listSource = []
    listTarget = []
    tragetCopy = np.asarray(copy.deepcopy(target).points)
    sourceCopy = np.asarray(copy.deepcopy(source).points)
    
    # calculate the dist between all points and copy.
    M = np.asarray(ot.dist(sourceCopy, tragetCopy))
    M_result = copy.deepcopy(M)
    for i in range(len(sourceCopy)):
        for j in range(len(tragetCopy)):
            if M_result[i,j]<=distanceThreshold:
                M_result[i,j]=1
                listSource.append(i)
                listTarget.append(j)
            else:
                M_result[i,j]=0
    return M_result, listSource, listTarget


# For loading the point clouds : return -
# (original source , original target , voxel down source , voxel down target , FPFH source , FPFH target).
def prepare_dataset(voxel_size, source_path, target_path, trans_init, method, VISUALIZATION, farthest_size=0.03, gamma_21=0.27, gamma_32=0.12,pathSaveImg="images/eth/",directoryName=""):
    
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path))
    
    source_down_c = preprocess_point_cloud_for_test(source, voxel_size * 4)
    target_down_c =  preprocess_point_cloud_for_test(target, voxel_size * 4)
    M_result, listSource, listTarget = findCorrZeroOne(source_down_c, target_down_c, 0.1001)
    
    if VISUALIZATION:
        draw_registration_result(source, target, np.identity(4), "Target Matching")
    savePCDS(source, target, "Target_Matching", pathSaveImg, np.identity(4),directoryName)
    
    source.transform(trans_init)
    source_down_c.transform(trans_init)
    
    if VISUALIZATION:
        draw_registration_result(source, target, np.identity(4), "Problem")
    savePCDS(source, target, "Problem", pathSaveImg, np.identity(4),directoryName)    
        
    if method == "voxel":
        source_down, source_fpfh = preprocess_point_cloud_voxel(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud_voxel(target, voxel_size)
        
        return source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget

    if method == "keypoints":
        source_down = preprocess_point_cloud_for_test(source, voxel_size)
        target_down =  preprocess_point_cloud_for_test(target, voxel_size) 
        
        # Source and target keypoint by iss method  
        source_key = o3d.geometry.keypoint.compute_iss_keypoints(source, gamma_21=gamma_21, gamma_32=gamma_32)
        target_key = o3d.geometry.keypoint.compute_iss_keypoints(target, gamma_21=gamma_21, gamma_32=gamma_32)
                
        # Source and target keypoint correspondence
        source_keyPointArr = np.asarray(source_key.points)
        target_keyPointArr = np.asarray(target_key.points)
        
        # Find the keypoints from voxel
        source_VoxelIdx, target_VoxelIdx, source_down_key, target_down_key = findVoxelCorrIdx(source_down.points, target_down.points, source_keyPointArr, target_keyPointArr)
            
        source_fpfh = preprocess_point_cloud_keypoint(source_down_key, voxel_size)
        target_fpfh = preprocess_point_cloud_keypoint(target_down_key, voxel_size)
        
        # source_fpfh_arr = np.asarray(source_fpfh.data)
        # target_fpfh_arr = np.asarray(target_fpfh.data)
        # source_fpfh_arr = source_fpfh_arr[:, source_VoxelIdx]
        # target_fpfh_arr = target_fpfh_arr[:, target_VoxelIdx]
        # source_fpfh_n = o3d.pipelines.registration.Feature()
        # target_fpfh_n = o3d.pipelines.registration.Feature()
        # source_fpfh_n.data = source_fpfh_arr
        # target_fpfh_n.data = target_fpfh_arr
    
        return source, target, source_down_key, target_down_key, source_key, target_key, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget

    if method == "fartest_point":
        source_down, source_fpfh = preprocess_point_cloud_fartest_point(source, voxel_size,farthest_size)
        target_down, target_fpfh = preprocess_point_cloud_fartest_point(target, voxel_size,farthest_size)
        return source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget


def findVoxelCorrIdx(realS, realT, keyS, keyT):
    
    realSidx = []
    realTidx = []
    realSarr = np.asarray(copy.deepcopy(realS))
    realTarr = np.asarray(copy.deepcopy(realT))
    
    # For each keypoint look for in the voxel - for source
    for key in keyS:
        flag = 0
        for i in range((realSarr.shape[0])):
            if key[0] == realSarr[i, 0] and key[1] == realSarr[i, 1] and key[2] == realSarr[i, 2]:
                realSidx.append(i)
                flag = 1
                break
        if flag == 0:
            arr_to_add = np.array([key[0], key[1], key[2]])
            realSarr = np.vstack([realSarr, arr_to_add])
            realSidx.append(realSarr.shape[0] - 1)
    
    # For each keypoint look for in the voxel - for target
    for key in keyT:
        flag = 0
        for i in range((realTarr.shape[0])):
            if key[0] == realTarr[i, 0] and key[1] == realTarr[i, 1] and key[2] == realTarr[i, 2]:
                realTidx.append(i)
                flag = 1
                break
        if flag == 0:
            arr_to_add = np.array([key[0], key[1], key[2]])
            realTarr = np.vstack([realTarr, arr_to_add])
            realTidx.append(realTarr.shape[0]-1)

    pcdA = o3d.geometry.PointCloud()
    pcdB = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(realSarr)
    pcdB.points = o3d.utility.Vector3dVector(realTarr)
    
    return realSidx, realTidx, pcdA, pcdB
    
######################################################################
##################        RANSAC_ICP_Refernce       ##################
######################################################################

# Run global regestration by RANSAC
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
    distance_threshold = 0.1001
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
    ransac_n = 3
    checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
    seed = 1

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, True,distance_threshold, estimation_method, ransac_n, checkers, criteria, seed)
    return result

# Run local regestration by icp with transformation result from global regestretion such as RANSAC.
def refine_registration(source, target, result_ransac):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(True), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 1000))
    return result

######################################################################
##################  Sinkhorn_RANSAC_ICP_Refernce    ##################
######################################################################

# Run global regestration by RANSAC
def execute_global_registration_with_corr(source_down, target_down, corr):
    distance_threshold = 0.1001
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
    ransac_n = 3
    checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999)
    seed = 1
    corr_size = 0.1001
    corr = o3d.utility.Vector2iVector(corr)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_down, target_down, corr, corr_size, estimation_method, ransac_n, checkers, criteria, seed)
    return result

# Run local regestration by icp with transformation result from global regestretion such as RANSAC.
def refine_registration_sinkhorn_ransac(source, target, result_ransac):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1200))
    return result

# Run local regestration by icp with transformation result from global regestretion such as RANSAC.
def refine_registration_sinkhorn_svd_icp(source, target, res):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, res,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1200))
    return result

def farthest_point(pcd, numOfPoints):
    pcdArr = np.asarray(pcd.points)
    point_idx = farthest_point_sampler(torch.reshape(torch.tensor(pcdArr), (1, pcdArr.shape[0], pcdArr.shape[1])), numOfPoints, 0)
    pcdResArr = pcdArr[point_idx[0]]
    pcdRes = o3d.geometry.PointCloud()
    pcdRes.points = o3d.utility.Vector3dVector(pcdResArr)
    return pcdRes

######################################################################
##################       Sinkhorn_SVD_Refernce      ##################
######################################################################