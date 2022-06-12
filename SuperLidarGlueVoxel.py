import torch
import open3d as o3d
import ETHDataSet as ETH
import Utilities as F
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import GatSinkhorn as GATS

VERBOSE = True
VISUALIZATION = True
voxel_size = 1

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

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


def loss2(all_matches, scores):
    # check if indexed correctly
    loss = []
    for i in range(all_matches.shape[1]):
      for j in range(all_matches.shape[2]):
        if all_matches[0][i][j] == 1:
            loss.append(-torch.log(scores[0][i][j].exp() )) # check batch size == 1 ?
    # for p0 in unmatched0:
    #     loss += -torch.log(scores[0][p0][-1])
    # for p1 in unmatched1:
    #     loss += -torch.log(scores[0][-1][p1])
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
        for batch_idx, (fpfhSourceTargetConcatenate, source_down, target_down, edge_index_self, edge_index_cross, sourceSize, targetSize, scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx) in enumerate(test_loader):
            
            if VERBOSE:
                print("\n================", batch_idx, "================\n")
            data = Data(x=fpfhSourceTargetConcatenate[0], edge_index=edge_index_self[0], edge_index2=edge_index_cross[0])
            data.x = torch.tensor(data.x, dtype=torch.float).to(device)
            optimizer.zero_grad()
            # sink = model(data, sourceSize ,targetSize)




            # source_arr = fpfhSourceTargetConcatenate[0:sourceSize, :]
            # target_arr = fpfhSourceTargetConcatenate[sourceSize: sourceSize + targetSize, :]

            # # ###############################################
            # scores = torch.einsum('bdn,bdm->bnm',source_arr.T, target_arr.T)
            # scores = scores.reshape(1,scores.shape[0], scores.shape[1])

            scores = [[0,0.14,0.4,1,0.23,0],
                      [0.9,-1,0.5,0.32,0.21, 0],
                      [0.42,1,0.25,0.6,0.3,0.7],
                      [0.8,0,0.55,0.25,1,1],
                      [0.8,0,0.55,0.25,1,1],
                      [0.05,0,0.15,0,-1,0],]
            scores = torch.tensor(scores)
            scores = scores.reshape(1, scores.shape[0], scores.shape[1])
            dustBin = torch.tensor(1.)
            num_iter = 1000  
            scores = F.log_optimal_transport(scores=scores, alpha=dustBin, iters=num_iter)
            
            
            
            # Get the matches with score above "match_threshold".
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = scores.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))            
            
            
            
            
            
            
            
            
            
            
            # outx=out[0, 0 : out.shape[1] - 1, 0 : out.shape[2] - 1]
            # max = np.unravel_index(torch.argmax(outx, axis=None), outx.shape)
            # print("max", max)
        
            # Save corr weights.
            # print("Correspondence set index values: ", max[0], max[1], scoreMatrix[0, max[0], max[1]], out[0, max[0], max[1]])
            # print("Num of corr", scoreMatrix.sum())
            # out1 = out.detach().numpy()
            # out1 = out1[0,0:out1.shape[1] - 1, 0:out1.shape[2] - 1]
            # scoreMatrix1 = scoreMatrix.detach().numpy()
            
            # Take number of top corr from sinkhorn result, take also the corr weights and print corr result.
            # corr_size = 500
            # corr = np.zeros((corr_size, 2))
            # corr_weights = np.zeros((corr_size, 1))
            # j = 0
            # sink[0, scoreMatrix.shape[0]-1, :] = 0
            # sink[0, :, scoreMatrix.shape[1]-1] = 0
            # while j < corr_size:
            #     max = np.unravel_index(np.argmax(sink, axis=None), sink.shape)
            #     corr[j][0], corr[j][1] = max[0], max[1]
            #     # Save corr weights.
            #     corr_weights[j] = sink[0, max[0], max[1]]  # Pn
            #     sink[0, max[0], :] = 0
            #     sink[0, :, max[1]] = 0
            #     j = j+1
            # print("Correspondence set index values: ", corr)

            # # Build numpy array for original points
            # source_arr = np.asarray(source_down.points)
            # target_arr = np.asarray(target_down.points)

            # # Take only the relevant indexes (without dust bin)
            # corr_values_source = source_arr[corr[:, 0].astype(int), :]  # Xn
            # corr_values_target = target_arr[corr[:, 1].astype(int), :]  # Yn

            # pcdS = o3d.geometry.PointCloud()
            # pcdS.points = o3d.utility.Vector3dVector(corr_values_source)
            # pcdT = o3d.geometry.PointCloud()
            # pcdT.points = o3d.utility.Vector3dVector(corr_values_target)
            # UR.draw_registration_result(pcdS, pcdT, np.identity(4), "debug")

            # Norm to sum equal to one for corr weights.
            # corr_weights = (corr_weights / np.sum(corr_weights))  # Pn norm

            # # Calc the mean of source and target point/FPFH with respect to points weight.
            # source_mean = np.sum(corr_values_source*corr_weights, axis=0)/np.sum(corr_weights)  # X0
            # target_mean = np.sum(corr_values_target*corr_weights, axis=0)/np.sum(corr_weights)  # Y0

            # # Calc the mean-reduced coordinate for Y and X
            # corr_values_source = corr_values_source-source_mean  # An
            # corr_values_target = corr_values_target-target_mean  # Bn

            # print(corr_values_source.shape, corr_values_target.shape, corr_weights.shape, source_mean.shape, target_mean.shape)

            # loss_batch=loss2(scoreMatrix,out)
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
            
            # loss_batch /= (out.shape[0] - 1)
            # loss_batch = torch.tensor(loss_batch,requires_grad=True)
            
            # if VERBOSE:
                # print("loss batch = ", loss_batch)
            
            # loss_batch.backward()
            # optimizer.step()
            
            # loss_epoch += loss_batch
        
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









        