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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    model.train()
    for epoch in range(num_epochs):
        # Display pcds and deatils for each problem.
        for batch_idx, (fpfhSourceTargetConcatenate, edge_index_self, edge_index_cross, sourceSize, targetSize, scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx) in enumerate(test_loader):
            
            if VERBOSE:
                print("\n================", batch_idx, "================\n")
            data = Data(x=fpfhSourceTargetConcatenate[0], edge_index=edge_index_self[0], edge_index2=edge_index_cross[0])
            data.x = torch.tensor(data.x, dtype=torch.float).to(device)
                        
            model.train()
            optimizer.zero_grad()
            out = model(data, sourceSize ,targetSize)
            print("Z", out)


            # loss.backward()
            optimizer.step()
        