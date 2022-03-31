import torch
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
    for batch_idx, (source, target, a, b, source_fpfh, target_fpfh, source_keyPointsIdx, target_keyPointsIdx, source_RealIdx, target_RealIdx, overlap, M, scoreMatrix) in enumerate(test_loader):
        print("================", batch_idx, "==============")
        print("source", len(source))
        print("target", len(target))
        print("source_fpfh", len(source_fpfh))
        print("target_fpfh", len(target_fpfh))
        print("source_keyPointsIdx", len(source_keyPointsIdx))
        print("target_keyPointsIdx", len(target_keyPointsIdx))
        print("source_RealIdx", len(source_RealIdx))
        print("target_RealIdx", len(target_RealIdx))
        print("overlap", overlap)
        print("M", M)
        print("a", a)
        print("b", b)
        # F.draw_registration_result(source, target, np.identity(4))
