import numpy as np
import open3d as o3d
import copy
import pandas as pd
import ot
import torch
from dgl.geometry import farthest_point_sampler


######################################################################
##################         General Functions        ##################
######################################################################

# # Normlize data between 0 to 1.
# def normlizeToOne(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))
# # Calculate score from invers translation marix.
# def calcScore(inversMatrix, resMatrix):
#     return 1-np.sum(abs((Normalize(inversMatrix)-Normalize(resMatrix))))/16

# Method get the data from global file for POC
def get_data_global_POC(directory):
    headers = ['id', 'source', 'target', 'overlap', 't1', 't2',
               't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
    read_file = pd.read_csv('Datasets/eth/' + directory +
                            '_global.txt', sep=" ", header=0, names=headers)
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


def farthest_point(pcd, numOfPoints):
    pcdArr = np.asarray(pcd.points)
    point_idx = farthest_point_sampler(torch.reshape(
        torch.tensor(pcdArr), (1, pcdArr.shape[0], pcdArr.shape[1])), numOfPoints, 0)
    pcdResArr = pcdArr[point_idx[0]]
    pcdRes = o3d.geometry.PointCloud()
    pcdRes.points = o3d.utility.Vector3dVector(pcdResArr)
    return pcdRes


# Method get the data


def get_data_global(directory):
    headers = ['id', 'source', 'target', 'overlap', 't1', 't2',
               't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
    read_file = pd.read_csv('Datasets/eth/' + directory +
                            '_global.txt', sep=" ", header=0, names=headers)
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
    o3d.visualization.draw_geometries(
        [source_temp, target_temp], title, 1280, 720)


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud(pcd, voxel_size):
    radius_normal = voxel_size * 5
    radius_feature = voxel_size * 10
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = farthest_point(pcd, int(0.03 * np.asarray(pcd.points).shape[0]))
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=250))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
    return pcd_down, pcd_fpfh

# For pre prossecing the point cloud - compute voxel.


def preprocess_point_cloud_voxel(pcd, voxel_size):
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
            if M_result[i, j] <= distanceThreshold:
                M_result[i, j] = 1
                listSource.append(i)
                listTarget.append(j)
            else:
                M_result[i, j] = 0
    return M_result, listSource, listTarget

######################################################################
##################        RANSAC_ICP_Refernce       ##################
######################################################################


# For loading the point clouds : return -
# (original source , original target , voxel down source , voxel down target , FPFH source , FPFH target).
def prepare_dataset(voxel_size, source_path, target_path, trans_init):
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path))
    # source_farthest = farthest_point(source, 500)
    # target_farthest = farthest_point(target, 500)
    # draw_registration_result(source_farthest, target_farthest, np.identity(4), "farthest_point")
    source_down_c = preprocess_point_cloud_voxel(source, voxel_size * 10)
    target_down_c = preprocess_point_cloud_voxel(target, voxel_size * 10)
    M_result, listSource, listTarget = findCorrZeroOne(
        source_down_c, target_down_c, 0.1001)
    draw_registration_result(source, target, np.identity(4), "Target Matching")
    source.transform(trans_init)
    source_down_c.transform(trans_init)
    draw_registration_result(source, target, np.identity(4), "Problem")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget

# Run global regestration by RANSAC


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
    distance_threshold = 0.1001
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        True)
    ransac_n = 3
    checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        1000000, 0.999)
    seed = 1

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold, estimation_method, ransac_n, checkers, criteria, seed)
    return result

# Run local regestration by icp with transformation result from global regestretion such as RANSAC.


def refine_registration(source, target, result_ransac):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(
        True), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
    return result

######################################################################
##################  Sinkhorn_RANSAC_ICP_Refernce    ##################
######################################################################

# For pre prossecing the point cloud - make voxel and compute FPFH.


def preprocess_point_cloud_sinkhorn_ransac(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 5
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=250))
    radius_feature = voxel_size * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
    return pcd_down, pcd_fpfh


# For loading the point clouds : return -
# (original source , original target , voxel down source , voxel down target , FPFH source , FPFH target).
def prepare_dataset_sinkhorn_ransac(voxel_size, source_path, target_path, trans_init):
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path))
    draw_registration_result(source, target, np.identity(4), "Target Matching")
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4), "Problem")
    source_down, source_fpfh = preprocess_point_cloud_sinkhorn_ransac(
        source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud_sinkhorn_ransac(
        target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# Run global regestration by RANSAC
def execute_global_registration_with_corr(source_down, target_down, corr):
    distance_threshold = 0.1001
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        True)
    ransac_n = 3
    checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        1000000, 0.9999)
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


def refine_registration_sinkhorn_svd(source, target, res):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, res,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1200))
    return result

######################################################################
##################       Sinkhorn_SVD_Refernce      ##################
######################################################################
