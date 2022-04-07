import pandas as pd
import numpy as np
import copy
import open3d as o3d
from torch.utils.data import Dataset
import Utilities as F


class ETHDataset(Dataset):

    #Init the ETH Dataset
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
        
        return np.asarray(source_.points), np.asarray(target_.points), overlap, M

    # Method to get all the problems from global file
    def get_data_global(self, directory):
        
        headers = ['id', 'source', 'target', 'overlap', 't1', 't2','t3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
        read_file = pd.read_csv('eth/' + directory + '_global.txt', sep=" ", header=0, names=headers)
        read_file.to_csv('eth/' + directory + '_global.csv', sep=',')
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
               = F.preprocessing(self.pcd_list.at[idx, 'source'], self.pcd_list.at[idx, 'target'], self.pcd_list.at[idx, 'overlap'], M)   
        return self.pcd_list.at[idx, 'source'], self.pcd_list.at[idx, 'target'], self.pcd_list.at[idx, 'overlap'], M

    # Method to get source and target as PCDs
    def prepare_item(self, source_path, target_path, trans_init):
        
        source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
        target = copy.deepcopy(o3d.io.read_point_cloud(target_path))       
        return source, target
