import numpy as np
import open3d as o3d
import copy
import pandas as pd
import UtilitiesReference as UR

if __name__ == '__main__':
    directories = ['apartment', 'hauptgebaude', 'wood_autumn', 'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs',  'plain']
    results = []
    avg_result_datasets = []
    iter_dataset = 0
    sum = 0
    sum_datasets = 0
    for directory in directories:
        sources, targets, overlaps, translation_M = UR.get_data_global(directory)
        results.append([])
        sum = 0
        for i in range(len(sources)):
            print("directory:" + directory + ", iter:" + str(i + 1) + "/" + str(len(sources)))
            # Save path for source & target pcd.
            source_path = 'Datasets/eth/' + directory + '/' + sources[i]
            target_path = 'Datasets/eth/' + directory + '/' + targets[i]

            # Init voxel for less num of point clouds.
            voxel_size = 0.1  # means 5cm for this dataset ?

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_fpfh, target_fpfh = UR.prepare_dataset(voxel_size, source_path, target_path, translation_M[i])

            # Execute global registration by RANSAC and FPFH , print the result and the correspondence point set .
            result_ransac = UR.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)

            # Execute local registration by ICP , Originals pcd and the global registration transformation result,
            # print the result and the correspondence point set .
            result_icp = UR.refine_registration(source, target, result_ransac)
            
            # If the result is not bigger then the overlap else if result bigger than the overlap BUT STILL MISMATCH
            if(result_icp.fitness > overlaps[i]):
                fitness = 2 - (result_icp.fitness / overlaps[i])
            else: 
                fitness = (result_icp.fitness / overlaps[i])
            results[iter_dataset].append([sources[i] + " " + targets[i], fitness])
            sum += results[iter_dataset][i][1]
            
            print(results[iter_dataset][i][0], "fitness =", fitness)
            print("avarage score until now =", sum / len(results[iter_dataset]))
            UR.draw_registration_result(source, target, result_icp.transformation, "ICP result")
        
        avg_result_datasets.append([directory, sum / len(results[iter_dataset])])
        print("avg result of dataset", directory, "is", avg_result_datasets[iter_dataset][1])
        sum_datasets += avg_result_datasets[iter_dataset][1]
        iter_dataset += 1
    
    total_avg = sum_datasets / len(avg_result_datasets)
    print()
    for i in range(len(avg_result_datasets)):
        print(avg_result_datasets[i][0], '\'s score: ', avg_result_datasets[i][1])
    print()
    print("total avarage = ", total_avg)
