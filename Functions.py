import sys
import numpy as np
import open3d as o3d
import copy
import ot


def NormalizeRow(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def findCorr(source, target, distanceThreshold):
    # prepare list and copy data.
    listSource = []
    listTarget = []
    tragetCopy = np.asarray(copy.deepcopy(target))
    sourceCopy = np.asarray(copy.deepcopy(source))
    # calculate the dist between all points and copy.
    M = np.asarray(ot.dist(sourceCopy, tragetCopy))
    M_result = copy.deepcopy(M)
    # save the maximum number correspondence to find.
    maxNumOfCorrToFind = min(M.shape[0], M.shape[1])
    # loop until reach maximum number if correspondence to find or arrive to non match points.
    while(maxNumOfCorrToFind > 0):
        # Find index of minimum value from dist matrix
        minNumIdx = np.unravel_index(np.argmin(M, axis=None), M.shape)
        # arrive to non match points - because from here distance always bigger from threshold.
        if distanceThreshold < M[minNumIdx[0], minNumIdx[1]]:
            break
        # set the index to max number for found next one.
        M[minNumIdx[0], minNumIdx[1]] = sys.maxsize
        # if the point not already corr with other point.
        if minNumIdx[0] not in listSource and minNumIdx[1] not in listTarget:
            M_result[minNumIdx[0], :] = NormalizeRow(M_result[minNumIdx[0], :])
            listSource.append(minNumIdx[0])
            listTarget.append(minNumIdx[1])
            maxNumOfCorrToFind -= 1

    M_result[not listSource, :] = -1
    res = np.ones((M_result.shape))*-1
    res[listSource, :] = M_result[listSource, :]
    return res, listSource, listTarget


def findRealCorrIdx(realS, realT, keyS, keyT, idxKeyS, idxKeyT):
    realSidx = []
    realTidx = []
    realSarr = np.asarray(copy.deepcopy(realS))
    realTarr = np.asarray(copy.deepcopy(realT))
    for key in idxKeyS:
        for i in range((realSarr.shape[0])):
            if keyS[key, 0] == realSarr[i, 0] and keyS[key, 1] == realSarr[i, 1] and keyS[key, 2] == realSarr[i, 2]:
                realSidx.append(i)
                break
    for key in idxKeyT:
        for i in range((realTarr.shape[0])):
            if keyT[key, 0] == realTarr[i, 0] and keyT[key, 1] == realTarr[i, 1] and keyT[key, 2] == realTarr[i, 2]:
                realTidx.append(i)
                break
    return realSidx, realTidx


def findVoxelCorrIdx(realS, realT, keyS, keyT, idxKeyS, idxKeyT):
    realSidx = []
    realTidx = []
    realSarr = np.asarray(copy.deepcopy(realS))
    realTarr = np.asarray(copy.deepcopy(realT))
    for key in idxKeyS:
        flag = 0
        for i in range((realSarr.shape[0])):
            if keyS[key, 0] == realSarr[i, 0] and keyS[key, 1] == realSarr[i, 1] and keyS[key, 2] == realSarr[i, 2]:
                realSidx.append(i)
                flag = 1
                break
        if flag == 0:
            realSarr = np.append(realSarr, keyS[key, :])
            realSidx.append(realSarr.shape[0]-1)

    for key in idxKeyT:
        flag = 0
        for i in range((realTarr.shape[0])):
            if keyT[key, 0] == realTarr[i, 0] and keyT[key, 1] == realTarr[i, 1] and keyT[key, 2] == realTarr[i, 2]:
                realTidx.append(i)
                flag = 1
                break
        if flag == 0:
            realTarr = np.append(realTarr, keyT[key, :])
            realTidx.append(realTarr.shape[0]-1)

    pcdA = o3d.geometry.PointCloud()
    pcdB = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(realSarr)
    pcdB.points = o3d.utility.Vector3dVector(realTarr)
    return realSidx, realTidx, pcdA, pcdB


def draw_registration_result(source, target, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp], width=1280, height=720)


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud(source, target, voxel_size):
    radius_normal = 5*voxel_size
    radius_feature = 10*voxel_size
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    pcd_fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=600))
    pcd_fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=600))
    return pcd_fpfh_source, pcd_fpfh_target
