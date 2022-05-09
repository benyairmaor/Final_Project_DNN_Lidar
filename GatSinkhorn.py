import torch
import numpy as np
import ot
import open3d as o3d
import Utilities as U
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.dustBin = torch.nn.Parameter(torch.tensor(1.))
        # self.hid = hid
        # self.in_head = in_head
        # self.out_head = out_head

        # self.conv1 = GATv2Conv(num_features, self.hid,
        #                        heads=self.in_head, dropout=0.6)
        # self.conv2 = GATv2Conv(num_features, self.hid,
        #                        heads=self.in_head, dropout=0.6)
        # self.conv3 = GATv2Conv(self.hid*self.in_head, num_classes,
        #                        concat=False, heads=self.out_head, dropout=0.6)
        # self.sinkhorn = log_optimal_transport

    def forward(self, data, sourceSize, targetSize):
        x, edge_index, edge_index2 = data.x, data.edge_index, data.edge_index2
        ###############################################
        # print(x)
        # for i in range(len(x)):
        #     x[i, :] = (x[i, :] - torch.min(x[i, :])) / \
        #         (torch.max(x[i, :]) - torch.min(x[i, :]))
        # print(x)
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        # print("x max: ", x.max(), " x min: ", x.min())

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

        source_arr = problem[0:sourceSize, :]
        target_arr = problem[sourceSize: sourceSize + targetSize, :]

        ###############################################
        scores = torch.einsum('dn,dm->nm', source_arr.T, target_arr.T)
        scores = scores / 33**.5
        # scores = scores.reshape(1, scores.shape[0], scores.shape[1])
        # print(scores.shape)
        # dustBin = torch.nn.Parameter(torch.tensor(1.))
        # self.register_parameter('dustBin', dustBin)

        # Print loss matrix shape
        # print("Loss matrix scores shape : ", scores.size())

        num_iter = 1000
        scores = U.sinkhorn(scores, eps=1e-9, maxiters=1000)

        # # Get the matches with score above "match_threshold".
        # max0, max1 = scores[:-1, :-1].max(1), scores[:-1, :-1].max(0)
        # indices0, indices1 = max0.indices, max1.indices
        # mutual0 = arange_like(indices0, 1)[
        #     None] == indices1.gather(1, indices0)
        # mutual1 = arange_like(indices1, 1)[
        #     None] == indices0.gather(1, indices1)
        # # print("\n\n################\n\n")
        # # print("\nmax0 :\n", max0)
        # # print("\n\n################\n\n")
        # # print("\n\n################\n\n")
        # # print("\nmutual0 :\n", mutual0)
        # # print("\n\n################\n\n")
        # zero = scores.new_tensor(0)
        # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        # mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # valid0 = mutual0 & (mscores0 > 0)
        # valid1 = mutual1 & valid0.gather(1, indices1)
        # indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        # indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        # # print("\n\n################\n\n")
        # # print("\nmscores0 :\n", mscores0)
        # # print("\n\n################\n\n")
        # # print("\n\n################\n\n")
        # # print("\nindices0 :\n", indices0)
        # # print("\n\n################\n\n")

        return scores, None, None


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
