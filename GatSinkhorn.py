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
        # self.hid = hid
        # self.in_head = in_head
        # self.out_head = out_head
        
        # self.conv1 = GATv2Conv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        # self.conv2 = GATv2Conv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        # self.conv3 = GATv2Conv(self.hid*self.in_head, num_classes, concat=False, heads=self.out_head, dropout=0.6)
        self.dustBin = torch.nn.Parameter(torch.tensor(1.))
        # self.sinkhorn = U.log_optimal_transport()

    def forward(self, data, sourceSize ,targetSize):
        x, edge_index, edge_index2 = data.x, data.edge_index, data.edge_index2
        
        ###############################################
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        print("x max: ",x.max()," x min: ",x.min())

        # for i in range(4):        
        #     x = F.dropout(x, p=0.6, training=self.training)
        #     x = self.conv1(x, edge_index)
        #     x = F.elu(x)
        #     x = F.dropout(x, p=0.6, training=self.training)
        #     x = self.conv2(x, edge_index2)
        #     x = F.log_softmax(x, dim=1)
        # x = self.conv3(x, edge_index)
        # x = F.log_softmax(x, dim=1)

        problem = x
        ###############################################
        # problem = problem / 33**.5
        source_arr = problem[0:sourceSize, :]
        target_arr = problem[sourceSize: sourceSize + targetSize, :]

        ###############################################
        scores = torch.einsum('dn,dm->nm',source_arr.T, target_arr.T)
        scores = scores.reshape(1,scores.shape[0], scores.shape[1])
        print(scores.shape)
        print("dustBin", self.dustBin)
        # dustBin = torch.nn.Parameter(torch.tensor(1.))
        # self.register_parameter('dustBin', dustBin)

        # Print loss matrix shape
        # print("Loss matrix scores shape : ", scores.size()) 

        num_iter = 100
        Z = U.log_optimal_transport(scores=scores, alpha=self.dustBin, iters=num_iter)

        return Z