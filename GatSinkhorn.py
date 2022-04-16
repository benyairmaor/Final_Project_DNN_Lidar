import torch
import numpy as np
import ot
import open3d as o3d
import Utilities as U
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hid = 8, in_head = 8, out_head = 1):
        super(GAT, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        
        self.conv1 = GATv2Conv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATv2Conv(self.hid*self.in_head, num_classes, concat=False, heads=self.out_head, dropout=0.6)

    def forward(self, data, sourceSize ,targetSize):
        x, edge_index, edge_index2 = data.x, data.edge_index, data.edge_index2

        for i in range(4):        
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index2)
            x = F.log_softmax(x, dim=1)

        problem = x.detach().numpy()
        source_arr = problem[0:sourceSize, :]
        target_arr = problem[sourceSize: sourceSize + targetSize, :]
        scores = np.asarray(ot.dist(source_arr, target_arr))
        # # Prepare dust bin for loss matrix M.
        # row_to_be_added = np.zeros(((target_arr.shape[0])))
        # column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
        # scores = np.vstack([scores, row_to_be_added])
        # scores = np.vstack([scores.T, column_to_be_added])
        # scores = scores.T

        scores.shape = (1, scores.shape[0], scores.shape[1])
        scores = torch.from_numpy(scores)

        dustBin = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('dustBin', dustBin)

        # Print loss matrix shape
        # print("Loss matrix scores shape : ", scores.size()) 
        num_iter = 100

        Z =U.log_optimal_transport(scores=scores, alpha=self.dustBin, iters=num_iter)

        return Z