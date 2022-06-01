import torch
import open3d as o3d
import ETHDataSet as ETH
import Utilities as F
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import GatSinkhorn as GATS

VERBOSE = True
VISUALIZATION = False
voxel_size = 2


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def loss2(all_matches, scores):
    loss = []
    for i in range(all_matches.shape[1]):
        for j in range(all_matches.shape[2]):
            if all_matches[0][i][j] == 1:
                # check batch size == 1 ?
                loss.append(-torch.log(scores[0][i][j].exp()))
    loss_mean = torch.mean(torch.stack(loss))
    loss_mean = torch.reshape(loss_mean, (1, -1))
    return loss_mean


def loss3(all_matches, indices):
    loss = []
    z = torch.tensor(0., requires_grad=True)
    o = torch.tensor(1., requires_grad=True)
    for i in range(indices.shape[1]):
        if(indices[0, i] != -1 and all_matches[0][i][indices[0, i]] == 1):
            loss.append(z)
        else:
            loss.append(o)
    loss_mean = torch.mean(torch.stack(loss))
    loss_mean = torch.reshape(loss_mean, (1, -1))
    return loss_mean


if __name__ == '__main__':
    ################################################### Data Loader #################################################

    num_epochs = 5
    # Load the ETH dataset
    ETH_dataset = ETH.ETHDataset("apartment")

    # Split the data to train test
    train_size = int(len(ETH_dataset) * 0.8)
    test_size = len(ETH_dataset) - int(len(ETH_dataset) * 0.8)
    #
    train_size = 1
    test_size = len(ETH_dataset) - 1
    train_set, test_set = torch.utils.data.random_split(
        ETH_dataset, [train_size, test_size])

    # TrainLoader, 80% of the data
    train_loader = DataLoader(train_set, batch_size=1,
                              num_workers=0, shuffle=False)

    #  TestLoader, 20% of the data
    test_loader = DataLoader(test_set, batch_size=1,
                             num_workers=0, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    model = GATS.GAT(33, 33, hid=33, in_head=8,
                     out_head=1).to(device).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(num_epochs):
        # Display pcds and deatils for each problem.
        loss_epoch = torch.tensor(0.)
        if VERBOSE:
            print("\n================", epoch, "================\n")
        for batch_idx, (fpfhSourceTargetConcatenate, source_down, target_down, edge_index_self, edge_index_cross, sourceSize, targetSize, scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx) in enumerate(test_loader):

            if VERBOSE:
                print("\n================", batch_idx, "================\n")
            data = Data(
                x=fpfhSourceTargetConcatenate[0], edge_index=edge_index_self[0], edge_index2=edge_index_cross[0])
            data.x = torch.tensor(data.x, dtype=torch.float).to(device)
            optimizer.zero_grad()
            sink, indices0, indices1 = model(data, sourceSize, targetSize)
            loss_batch = loss2(scoreMatrix, sink)
            if VERBOSE:
                print("loss batch = ", loss_batch)

            loss_batch.backward()
            optimizer.step()
            print("model.parameters(): ", model.parameters())
        #     loss_epoch += loss_batch

        # loss_epoch /= len(test_loader)

        # if VERBOSE:
        #     print("loss epoch = ", loss_epoch)

        # max = np.unravel_index(np.argmax(out1, axis=None), out1.shape)
        # min = np.unravel_index(np.argmin(out1, axis=None), out1.shape)
        # print(max[0], max[1], out1[max[0], max[1]])
        # print(min[0], min[1], out1[min[0], min[1]])
        # print("Z", out)
        # print("1", scoreMatrix1[0, max[0], max[1]])
        # print("0", scoreMatrix1[0, min[0], min[1]])
        # max = np.unravel_index(np.argmax(out1, axis=None), out1.shape)
        # min = np.unravel_index(np.argmin(out1, axis=None), out1.shape)
        # print(max[0], max[1], out1[max[0], max[1]])
        # print(min[0], min[1], out1[min[0], min[1]])
        # print("Z", out)
        # print("1", scoreMatrix1[0, max[0], max[1]])
        # print("0", scoreMatrix1[0, min[0], min[1]])
