from itertools import count
import torch
import numpy as np
import ot
import open3d as o3d
import Utilities as U
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
import torch.nn.functional as F


def normalizeEinsumResult(einsumMatrix, pcda, pcdb):
    for i in range(einsumMatrix.shape[0]):
        for j in range(einsumMatrix.shape[1]):
            einsumMatrix[i, j] = einsumMatrix[i, j] / \
                (torch.linalg.norm(pcda[i]) * torch.linalg.norm(pcdb[j]))


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hid=8, in_head=8, out_head=1):
        super(GAT, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        self.dustBin = torch.nn.Parameter(torch.tensor(1.))

        self.conv1 = GATv2Conv(num_features, self.hid,
                               heads=self.in_head, dropout=0)
        self.conv2 = GATv2Conv(self.hid*self.in_head,
                               self.hid, heads=self.in_head, dropout=0)
        self.conv3 = GATv2Conv(
            self.hid*self.in_head, num_classes, concat=False, heads=self.out_head, dropout=0)

    def forward(self, data, sourceSize, targetSize):
        x, edge_index, edge_index2 = data.x, data.edge_index, data.edge_index2
        ###############################################
        print(edge_index.shape)
        # x = F.softmax(x, dim=1)
        for i in range(1):
            if i == 0:
                # x = F.dropout(x, p=0.6, training=self.training)
                x = self.conv1(x, edge_index)
                # x = F.leaky_relu(x)
            # else:
                # x = F.dropout(x, p=0.6, training=self.training)
                # x = self.conv2(x, edge_index)
                # x = F.leaky_relu(x)
            # x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index2)
        #     # x = F.leaky_relu(x)
        #     # x = F.softmax(x, dim=1)
        # x = self.conv3(x, edge_index2)
        # x = F.leaky_relu(x)
        x = F.log_softmax(x, dim=1)

        problem = x
        ###############################################

        source_arr = problem[0:sourceSize, :]
        target_arr = problem[sourceSize: sourceSize + targetSize, :]
        # target_arr = source_arr
        source_fpfh_T_tensor = torch.tensor(source_arr.T)  # (33,x)
        target_fpfh_T_tensor = torch.tensor(target_arr.T)  # (33,y)

        ###############################################
        scores = torch.einsum(
            'dn,dm->nm', source_fpfh_T_tensor, target_fpfh_T_tensor)
        normalizeEinsumResult(
            scores, source_fpfh_T_tensor.T, target_fpfh_T_tensor.T)
        scores = scores.reshape(1, scores.shape[0], scores.shape[1])
        scores = log_optimal_transport(
            scores, alpha=self.dustBin, iters=1000)
        scores_tmp = scores.clone()
        for i in range(scores.shape[1]-1):
            scores_tmp[0, i, :scores_tmp.shape[2]-1] = scores_tmp[0, i,
                                                                  :scores_tmp.shape[2]-1]-torch.max(scores_tmp[0, i, :scores_tmp.shape[2]-1])

        print("self.dustBin / alphe =   ", self.dustBin)

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[
            None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[
            None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        howMachMinusOne = 0
        howMachMatching = 0
        test = 0
        counter = -1
        print(indices0.shape)
        for i in indices0[0]:
            counter = counter+1
            if i == -1:
                howMachMinusOne = howMachMinusOne+1
            else:
                if(i == counter):
                    test = test+1
                howMachMatching = howMachMatching+1
                # print("indices1[i], "  ", sm[i]", indices1[0, i], "  ", i)
                # print("test score matrix : ", sm[indices1[0, i], i])

        print("NUM MATCHING: ", howMachMatching)
        print("NUM OF MINUS ONE: ", howMachMinusOne)
        print("NUM OF TEST: ", test)
        return scores, indices0, indices1


def log_sinkhorn_iterations_t(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations_t(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
