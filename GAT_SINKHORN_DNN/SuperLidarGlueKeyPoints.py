import torch
import open3d as o3d
import ETHDataSet as ETH
import Utilities as F
import numpy as np
from torch.utils.data import DataLoader

VERBOSE = True

if __name__ == '__main__':
    ################################################### Data Loader #################################################

    # Load the ETH dataset
    ETH_dataset = ETH.ETHDataset("apartment")

    # Split the data to train test
    train_size = int(len(ETH_dataset) * 0.8)
    test_size = len(ETH_dataset) - int(len(ETH_dataset) * 0.8)
    train_set, test_set = torch.utils.data.random_split(ETH_dataset, [train_size, test_size])

    # TrainLoader, 80% of the data
    train_loader = DataLoader(train_set, batch_size=1, num_workers=0, shuffle=False)

    #  TestLoader, 20% of the data
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

    # # Display pcds and deatils for each problem.
    for batch_idx, (source, target, overlap, M) in enumerate(test_loader):
        
        # Simplify the data
        sourceArr = source.numpy()[0,:,:]
        targetArr = target.numpy()[0,:,:]
        source_ = o3d.geometry.PointCloud()
        target_ = o3d.geometry.PointCloud()
        source_.points = o3d.utility.Vector3dVector(sourceArr)
        target_.points = o3d.utility.Vector3dVector(targetArr)

        M = M.numpy()[0,:,:]
        overlap = overlap.numpy()[0]
        
        if VERBOSE:
            print("prepare item finished\n\n ================ The Problem Is: ================\n")
            print("source", source_)
            print("target", target_)
            print("overlap", overlap)
            print("M", M)

        if VERBOSE:
            print("\n================", batch_idx, "================\n")
        F.draw_registration_result(source_, target_, np.identity(4))

    ########################################### Preprocessing (Front End) #########################################

        # Source keypoint by iss method
        source_key = o3d.geometry.keypoint.compute_iss_keypoints(source_, gamma_21=0.27, gamma_32=0.12)
        
        if VERBOSE:
            print("source_key finished", source_key)

        # Target keypoint by iss method
        target_key = o3d.geometry.keypoint.compute_iss_keypoints(target_, gamma_21=0.27, gamma_32=0.12)
        
        if VERBOSE:
            print("target_key finished", target_key)
        
        # Source and target keypoint correspondence
        source_keyPointArr = np.asarray(source_key.points)
        target_keyPointArr = np.asarray(target_key.points)
        # Find the correspondence between the PCDs keypoints
        scoreMatrix, source_keyCorrIdx, target_keyCorrIdx = F.findCorr(source_keyPointArr, target_keyPointArr, 0.1001)
        
        if VERBOSE:
            print("\nkeypoints finished")
            print("source_keyCorrIdx", len(source_keyCorrIdx))
            print("target_keyCorrIdx", len(target_keyCorrIdx))
        
        # Downsampaling the PCD by voxel
        voxel_size = 0.2
        source_down = source_.voxel_down_sample(voxel_size)
        target_down = target_.voxel_down_sample(voxel_size)
        
        if VERBOSE:
            print("\nvoxel finished")
        
        # Find keypoints correspondence indecies in voxel
        source_VoxelIdx, target_VoxelIdx, source_down_key, target_down_key = F.findVoxelCorrIdx(source_down.points, target_down.points, source_keyPointArr, target_keyPointArr, source_keyCorrIdx, target_keyCorrIdx)
        
        if VERBOSE:
            print("indexies of the voxel world finished")
            print("source_down_key", source_down_key)
            print("target_down_key", target_down_key)
            print("source_VoxelIdx", len(source_VoxelIdx))
            print("target_VoxelIdx", len(target_VoxelIdx))
        
        # Calculate FPFH
        source_fpfh, target_fpfh = F.preprocess_point_cloud(source_down_key, target_down_key, voxel_size)
        
        if VERBOSE:
            print("\nfpfh finished")
            print("source_fpfh", source_fpfh)
            print("target_fpfh", target_fpfh)

        # Tramsform source
        source_.transform(M)
        
        # Visualize Keypoints
        source_key_arr = np.asarray(source_key.points)
        target_key_arr = np.asarray(target_key.points)
        source_key_corr_arr = np.zeros((len(source_keyCorrIdx), 3))
        target_key_corr_arr = np.zeros((len(target_keyCorrIdx), 3))
        
        counter = 0
        for i in source_keyCorrIdx:
            source_key_corr_arr[counter, :] = source_key_arr[i, :]
            counter += 1 
        
        counter = 0
        for i in target_keyCorrIdx:
            target_key_corr_arr[counter, :] = target_key_arr[i, :]
            counter += 1 
        
        pcdA = o3d.geometry.PointCloud()
        pcdB = o3d.geometry.PointCloud()
        pcdA.points = o3d.utility.Vector3dVector(source_key_arr)
        pcdB.points = o3d.utility.Vector3dVector(target_key_arr)
        F.draw_registration_result(pcdA, pcdB, np.identity(4))
        
        # Visualize Keypoints correspondence
        pcdC = o3d.geometry.PointCloud()
        pcdD = o3d.geometry.PointCloud()
        pcdC.points = o3d.utility.Vector3dVector(source_key_corr_arr)
        pcdD.points = o3d.utility.Vector3dVector(target_key_corr_arr)
        F.draw_registration_result(pcdC, pcdD, np.identity(4))
