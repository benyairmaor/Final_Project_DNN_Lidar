import torch
import open3d as o3d
import ETHDataSet as ETH
import Functions as F
import numpy as np
from torch.utils.data import DataLoader

if __name__ == '__main__':
    ################################################### Data Loader #################################################

    # Load the ETH dataset
    ETH_dataset = ETH.ETHDataset("apartment")

    # print("=============== get item ==================")
    # source, target, overlap, M = ETH_dataset[0]
    # print(source.points, target.points, overlap, M)

    # Split the data to train test
    train_size = int(len(ETH_dataset) * 0.8)
    test_size = len(ETH_dataset) - int(len(ETH_dataset) * 0.8)
    train_set, test_set = torch.utils.data.random_split(
        ETH_dataset, [train_size, test_size])

    # TrainLoader, 80% of the data
    train_loader = DataLoader(train_set, batch_size=1,
                              num_workers=0, shuffle=False)

    #  TestLoader, 20% of the data
    test_loader = DataLoader(test_set, batch_size=1,
                             num_workers=0, shuffle=False)

    # # Display pcds and deatils for each problem.
    for batch_idx, (source, target, source_down_key, target_down_key, source_fpfh, target_fpfh, source_key, target_key, source_keyCorrIdx, target_keyCorrIdx, source_RealIdx, target_RealIdx, overlap, M, scoreMatrix) in enumerate(test_loader):
        print("================", batch_idx, "==============")
        print("source", source.shape)
        print("target", target.shape)
        print("source_key", source_key.shape)
        print("target_key", target_key.shape)
        print("source_down_key", source_down_key.shape)
        print("target_down_key", target_down_key.shape)
        print("source_fpfh", source_fpfh.shape)
        print("target_fpfh", target_fpfh.shape)
        print("source_keyCorrIdx", len(source_keyCorrIdx))
        print("target_keyCorrIdx", len(target_keyCorrIdx))
        print("source_RealIdx", len(source_RealIdx))
        print("target_RealIdx", len(target_RealIdx))
        print("overlap", overlap)
        print("M", M)
        source_key_arr = source_key.numpy()[0,:,:]
        target_key_arr = target_key.numpy()[0,:,:]
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
        pcdC = o3d.geometry.PointCloud()
        pcdD = o3d.geometry.PointCloud()
        pcdC.points = o3d.utility.Vector3dVector(source_key_corr_arr)
        pcdD.points = o3d.utility.Vector3dVector(target_key_corr_arr)
        F.draw_registration_result(pcdC, pcdD, np.identity(4))
