import sys
import numpy as np
import open3d as o3d
import copy
import ot
import torch

############################### Normalize row of a matrix from 0-1 ###############################
def NormalizeRow(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



############################### Find the correspondence from source and target ###############################
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



############################### Find the indecies of keypoint in the whole PCD ###############################
def findRealCorrIdx(realS, realT, keyS, keyT, idxKeyS, idxKeyT):
    
    realSidx = []
    realTidx = []
    realSarr = np.asarray(copy.deepcopy(realS))
    realTarr = np.asarray(copy.deepcopy(realT))
    
    # For each keypoint look for the whole PCD - for source
    for key in idxKeyS:
        for i in range((realSarr.shape[0])):
            if keyS[key, 0] == realSarr[i, 0] and keyS[key, 1] == realSarr[i, 1] and keyS[key, 2] == realSarr[i, 2]:
                realSidx.append(i)
                break
    
    # For each keypoint look for the whole PCD - for target
    for key in idxKeyT:
        for i in range((realTarr.shape[0])):
            if keyT[key, 0] == realTarr[i, 0] and keyT[key, 1] == realTarr[i, 1] and keyT[key, 2] == realTarr[i, 2]:
                realTidx.append(i)
                break
    
    return realSidx, realTidx



############################### Find the indecies of keypoint in voxel and if not found add it ###############################
def findVoxelCorrIdx(realS, realT, keyS, keyT, idxKeyS, idxKeyT):
    
    realSidx = []
    realTidx = []
    realSarr = np.asarray(copy.deepcopy(realS))
    realTarr = np.asarray(copy.deepcopy(realT))
    
    # For each keypoint look for in the voxel - for source
    for key in idxKeyS:
        flag = 0
        for i in range((realSarr.shape[0])):
            if keyS[key, 0] == realSarr[i, 0] and keyS[key, 1] == realSarr[i, 1] and keyS[key, 2] == realSarr[i, 2]:
                realSidx.append(i)
                flag = 1
                break
        if flag == 0:
            arr_to_add = np.array([keyS[key, 0], keyS[key, 1], keyS[key, 2]])
            realSarr = np.vstack([realSarr, arr_to_add])
            realSidx.append(realSarr.shape[0]-1)
    
    # For each keypoint look for in the voxel - for target
    for key in idxKeyT:
        flag = 0
        for i in range((realTarr.shape[0])):
            if keyT[key, 0] == realTarr[i, 0] and keyT[key, 1] == realTarr[i, 1] and keyT[key, 2] == realTarr[i, 2]:
                realTidx.append(i)
                flag = 1
                break
        if flag == 0:
            arr_to_add = np.array([keyT[key, 0], keyT[key, 1], keyT[key, 2]])
            realTarr = np.vstack([realTarr, arr_to_add])
            realTidx.append(realTarr.shape[0]-1)

    pcdA = o3d.geometry.PointCloud()
    pcdB = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(realSarr)
    pcdB.points = o3d.utility.Vector3dVector(realTarr)
    
    return realSidx, realTidx, pcdA, pcdB



############################### For visualization ###############################
def draw_registration_result(source, target, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], width=1280, height=720)



############################### For voxel and compute FPFH ###############################
def preprocess_point_cloud(source, target, voxel_size):
    
    radius_normal = 5*voxel_size
    radius_feature = 10*voxel_size
    
    # Calculate normals
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    
    # Calculate FPFHs
    pcd_fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=600))
    pcd_fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=600))
    
    return pcd_fpfh_source, pcd_fpfh_target





VERBOSE = True
VISUALIZATION = True
voxel_size = 1
device = 'cpu'

def preprocessing(source, target, overlap, M):
    
    if VERBOSE:
        print("prepare item finished\n\n ================ The Problem Is: ================\n")
        print("source", source)
        print("target", target)
        print("overlap", overlap)
        print("M", M)

    if VISUALIZATION:
        draw_registration_result(source, target, np.identity(4))

########################################### Preprocessing (Front End) #########################################

    # Downsampaling the PCD by voxel
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down_arr = np.asarray(source_down.points)
    target_down_arr = np.asarray(target_down.points)

    # Find the correspondence between the PCDs voxel
    scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx = findCorr(source_down_arr, target_down_arr, 0.1001)
    
    if VERBOSE:
        print("\nvoxel finished")
        print("source_down", source_down)
        print("target_down", target_down)
        print("source_voxelCorrIdx", len(source_voxelCorrIdx))
        print("target_voxelCorrIdx", len(target_voxelCorrIdx))
    
    # Calculate FPFH
    source_fpfh, target_fpfh = preprocess_point_cloud(source_down, target_down, voxel_size)
    
    if VERBOSE:
        print("\nfpfh finished")
        print("source_fpfh", source_fpfh)
        print("target_fpfh", target_fpfh)

    # Tramsform source
    source.transform(M)
    
    if VISUALIZATION:
        source_key_corr_arr = np.zeros((len(source_down_arr), 3))
        target_key_corr_arr = np.zeros((len(target_down_arr), 3))
        
        counter = 0
        for i in source_voxelCorrIdx:
            source_key_corr_arr[counter, :] = source_down_arr[i, :]
            counter += 1 
        
        counter = 0
        for i in target_voxelCorrIdx:
            target_key_corr_arr[counter, :] = target_down_arr[i, :]
            counter += 1 
        
        pcdA = o3d.geometry.PointCloud()
        pcdB = o3d.geometry.PointCloud()
        pcdA.points = o3d.utility.Vector3dVector(source_down_arr)
        pcdB.points = o3d.utility.Vector3dVector(target_down_arr)
        draw_registration_result(pcdA, pcdB, np.identity(4))
        
        # Visualize voxel correspondence
        pcdC = o3d.geometry.PointCloud()
        pcdD = o3d.geometry.PointCloud()
        pcdC.points = o3d.utility.Vector3dVector(source_key_corr_arr)
        pcdD.points = o3d.utility.Vector3dVector(target_key_corr_arr)
        draw_registration_result(pcdC, pcdD, np.identity(4))

    source_fpfh_arr = np.asarray(source_fpfh.data).T
    target_fpfh_arr = np.asarray(target_fpfh.data).T
    
    if VERBOSE:
        print(source_fpfh_arr.shape, target_fpfh_arr.shape)

    fpfhSourceTargetConcatenate = np.concatenate((source_fpfh_arr, target_fpfh_arr), axis=0)
    fpfhSourceTargetConcatenate = torch.tensor(fpfhSourceTargetConcatenate)
    fpfhSourceTargetConcatenate = fpfhSourceTargetConcatenate.to(device)

    sourceSize = source_fpfh_arr.shape[0]
    targetSize = target_fpfh_arr.shape[0]

    selfMatrix = np.zeros((sourceSize + targetSize, sourceSize + targetSize))
    selfMatrix[0:sourceSize, 0:sourceSize] = 1
    selfMatrix[sourceSize:len(selfMatrix), sourceSize:len(selfMatrix)] = 1
    selfMatrix = torch.tensor(selfMatrix)

    for i in range(len(selfMatrix)):
        selfMatrix[i][i] = 0

    crossMatrix = np.ones((sourceSize + targetSize, sourceSize + targetSize))
    crossMatrix[0:sourceSize, 0:sourceSize] = 0
    crossMatrix[sourceSize:len(crossMatrix), sourceSize:len(crossMatrix)] = 0
    crossMatrix = torch.tensor(crossMatrix)

    # TODO: Don't know if needed
    # for i in range(len(crossMatrix)):
    #     crossMatrix[i][i] = 1

    edge_index_self = [[], []]
    for i in range(selfMatrix.shape[0]):
        for j in range(selfMatrix.shape[1]):
            if selfMatrix[i][j] == 1:
                edge_index_self[0].append(i)
                edge_index_self[1].append(j)
    
    edge_index_cross = [[], []]
    for i in range(crossMatrix.shape[0]):
        for j in range(crossMatrix.shape[1]):
            if crossMatrix[i][j] == 1:
                edge_index_cross[0].append(i)
                edge_index_cross[1].append(j)
    edge_index_self_ = torch.tensor(edge_index_self, dtype=torch.long)
    edge_index_cross_ = torch.tensor(edge_index_cross, dtype=torch.long)
    return fpfhSourceTargetConcatenate, edge_index_self_, edge_index_cross_, sourceSize, targetSize, scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)
 
 
def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)
 
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
 
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)
 
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
 
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z