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
voxel_size = 1

def loss(scoreMatrix,P):
    print(scoreMatrix.size(),P.size())
    matches=torch.tensor(0.)
    unmatches_a=torch.tensor(0.)
    unmatches_b=torch.tensor(0.)

    for i in range(scoreMatrix.size(1)):
        flag_row=False
        for j in range(scoreMatrix.size(2)):
            if scoreMatrix[0,i,j]==1:
                # print(torch.log(P[0,i,j]))
                # matches+=torch.log(P[0,i,j])
                matches+=(P[0,i,j])

                flag_row=True
        if flag_row == False:
            # unmatches_a+=torch.log(P[0,i,scoreMatrix.size(2)])
            unmatches_a+=(P[0,i,scoreMatrix.size(2)])
    for i in range(scoreMatrix.size(2)):
        flag_col=False
        for j in range(scoreMatrix.size(1)):
            if scoreMatrix[0,j,i]==1:
                flag_col=True
        if flag_col == False:
                # unmatches_b+=torch.log(P[0,scoreMatrix.size(1),i])
                unmatches_b+=(P[0,scoreMatrix.size(1),i])

    return torch.Tensor(-matches-unmatches_a-unmatches_b)



                


if __name__ == '__main__':
    ################################################### Data Loader #################################################

    num_epochs = 5
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
    # device = 'cpu'

    model = GATS.GAT(33, 33, hid=33, in_head=8, out_head=1).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(num_epochs):
        # Display pcds and deatils for each problem.
        loss_epoch = torch.tensor(0.)
        if VERBOSE:
            print("\n================", epoch, "================\n")
        for batch_idx, (fpfhSourceTargetConcatenate, edge_index_self, edge_index_cross, sourceSize, targetSize, scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx) in enumerate(test_loader):
            
            if VERBOSE:
                print("\n================", batch_idx, "================\n")
            data = Data(x=fpfhSourceTargetConcatenate[0], edge_index=edge_index_self[0], edge_index2=edge_index_cross[0])
            data.x = torch.tensor(data.x, dtype=torch.float).to(device)
                        
            optimizer.zero_grad()
            out = model(data, sourceSize ,targetSize)

            out1 = out.detach().numpy()
            out1 = out1[0,0:out1.shape[1] - 1, 0:out1.shape[2] - 1]
            scoreMatrix1 = scoreMatrix.detach().numpy()
            


            loss_batch=loss(scoreMatrix,out)
            # loss_batch = torch.tensor(0.)
            # for sourceIdx in range(out1.shape[0] - 1):
            #     targetIdx = np.argmax(out1[sourceIdx,:])
            #     if targetIdx == out1.shape[1]:
            #         if out1[sourceIdx, 0] != -1:
            #             loss_batch += 1
            #     else:
            #         if scoreMatrix1[0, sourceIdx, targetIdx] == -1:
            #             loss_batch += 1
            #         else:
            #             loss_batch += scoreMatrix1[0, sourceIdx, targetIdx]
            
            loss_batch /= (out1.shape[0] - 1)
            # loss_batch = torch.tensor(loss_batch,requires_grad=True)
            
            if VERBOSE:
                print("loss batch = ", loss_batch)
            
            loss_batch.backward()
            optimizer.step()
            
            loss_epoch += loss_batch
        
        loss_epoch /= len(test_loader)
        
        if VERBOSE:
            print("loss epoch = ", loss_epoch)
            

            
        # max = np.unravel_index(np.argmax(out1, axis=None), out1.shape)
        # min = np.unravel_index(np.argmin(out1, axis=None), out1.shape)
        # print(max[0], max[1], out1[max[0], max[1]])
        # print(min[0], min[1], out1[min[0], min[1]])
        # print("Z", out)
        # print("1", scoreMatrix1[0, max[0], max[1]])
        # print("0", scoreMatrix1[0, min[0], min[1]])


        