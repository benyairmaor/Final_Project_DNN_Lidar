import pandas as pd
import numpy as np
import copy
import open3d as o3d
from torch.utils.data import Dataset
import Utilities as F


class ETHDataset(Dataset):

    # Init the ETH Dataset
    def __init__(self, path_to_dir, transform=None, target_transform=None):

        self.pcd_list = self.get_data_global(path_to_dir)
        self.dir_name = path_to_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.pcd_list)

    def __getitem__(self, idx):

        # Extract problem from global file
        source, target, overlap, M = self.get_data(idx)
        source_path = 'eth/' + self.dir_name + '/' + source
        target_path = 'eth/' + self.dir_name + '/' + target
        source_, target_ = self.prepare_item(source_path, target_path, M)

        # TODO: Don't know if needed
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        fpfhSourceTargetConcatenate, edge_index_self, edge_index_cross, sourceSize, targetSize, scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx = F.preprocessing(
            source_, target_, overlap, M)
        return np.asarray(fpfhSourceTargetConcatenate), edge_index_self, edge_index_cross, sourceSize, targetSize, scoreMatrix, source_voxelCorrIdx, target_voxelCorrIdx

    # Method to get all the problems from global file
    def get_data_global(self, directory):

        headers = ['id', 'source', 'target', 'overlap', 't1', 't2',
                   't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
        read_file = pd.read_csv(
            'Datasets/eth/' + directory + '_global.txt', sep=" ", header=0, names=headers)
        read_file.to_csv('Datasets/eth/' + directory + '_global.csv', sep=',')
        read_file = pd.DataFrame(read_file, columns=headers)
        return read_file

    # Method to get one problem by index from all the problems
    def get_data(self, idx):

        M = np.zeros((4, 4))
        for i in range(1, 13):
            idx_row = int((i - 1) / 4)
            idx_col = (i - 1) % 4
            M[idx_row, idx_col] = self.pcd_list.at[idx, 't' + str(i)]
        M[3, :] = [0, 0, 0, 1]
        #################################################
        M = np.asarray([[-0.5754197841861329, 0.817372954385317,
                         -0.028169583003715, 11.778369303008173],
                        [-0.7611987839242382, -0.5478349625282469,
                         -0.34706377682485917, 14.264281414042465],
                        [-0.2991128270727379, -0.17826471123330384,
                         0.9374183747982869, 1.6731349336747363],
                        [0., 0., 0., 1.]])
        #################################################
        return self.pcd_list.at[idx, 'source'], self.pcd_list.at[idx, 'target'], self.pcd_list.at[idx, 'overlap'], M

    # Method to get source and target as PCDs
    def prepare_item(self, source_path, target_path, trans_init):
        #################################################
        source_path = "eth//hauptgebaude//PointCloud26.pcd"
        target_path = "eth//hauptgebaude//PointCloud27.pcd"
        ##################################################
        source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
        target = copy.deepcopy(o3d.io.read_point_cloud(target_path))
        return source, target
