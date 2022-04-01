import torch
import open3d as o3d
import ETHDataSet as ETH
import Utilities as F
import numpy as np
from torch.utils.data import DataLoader
# import GatSinkhorn as GATS

VERBOSE = True
VISUALIZATION = True
voxel_size = 1

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

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    # model = GATS().to(device)

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
        if VISUALIZATION:
            F.draw_registration_result(source_, target_, np.identity(4))

    ########################################### Preprocessing (Front End) #########################################

        # Downsampaling the PCD by voxel
        source_down = source_.voxel_down_sample(voxel_size)
        target_down = target_.voxel_down_sample(voxel_size)
        
        source_down_arr = np.asarray(source_down.points)
        target_down_arr = np.asarray(target_down.points)

        # Find the correspondence between the PCDs voxel
        scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx = F.findCorr(source_down_arr, target_down_arr, 0.1001)
        
        if VERBOSE:
            print("\nvoxel finished")
            print("source_down", source_down)
            print("target_down", target_down)
            print("source_voxelCorrIdx", len(source_voxelCorrIdx))
            print("target_voxelCorrIdx", len(target_voxelCorrIdx))
        
        # Calculate FPFH
        source_fpfh, target_fpfh = F.preprocess_point_cloud(source_down, target_down, voxel_size)
        
        if VERBOSE:
            print("\nfpfh finished")
            print("source_fpfh", source_fpfh)
            print("target_fpfh", target_fpfh)

        # Tramsform source
        source_.transform(M)
        
        if VISUALIZATION:
            # Visualize voxel
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
            F.draw_registration_result(pcdA, pcdB, np.identity(4))
            
            # Visualize voxel correspondence
            pcdC = o3d.geometry.PointCloud()
            pcdD = o3d.geometry.PointCloud()
            pcdC.points = o3d.utility.Vector3dVector(source_key_corr_arr)
            pcdD.points = o3d.utility.Vector3dVector(target_key_corr_arr)
            F.draw_registration_result(pcdC, pcdD, np.identity(4))

            source_fpfh_arr = np.asarray(source_fpfh.data).T
            target_fpfh_arr = np.asarray(target_fpfh.data).T
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

            crossMatrix = np.ones((sourceSize + targetSize, sourceSize + targetSize))
            crossMatrix[0:sourceSize, 0:sourceSize] = 0
            crossMatrix[sourceSize:len(crossMatrix), sourceSize:len(crossMatrix)] = 0
            crossMatrix = torch.tensor(crossMatrix)

            # TODO: Don't know if needed
            # for i in range(len(crossMatrix)):
            #     crossMatrix[i][i] = 1

            selfInput = torch.mm(selfMatrix, fpfhSourceTargetConcatenate)
            crossInput = torch.mm(crossMatrix, fpfhSourceTargetConcatenate)
            