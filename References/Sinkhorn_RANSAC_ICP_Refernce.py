import numpy as np
import open3d as o3d
import copy
import ot
import pandas as pd
from sqlalchemy import true
import UtilitiesReference as UR
VISUALIZATION = False
if __name__ == '__main__':

    directories = ['apartment', 'hauptgebaude', 'wood_autumn', 'gazebo_summer', 'gazebo_winter',
                   'wood_summer', 'stairs', 'plain']
    results = []
    avg_result_datasets = []
    iter_dataset = 0
    sum = 0
    sum_datasets = 0
    for directory in directories:
        sources, targets, overlaps, translation_M = UR.get_data_global(
            directory, true)
        results.append([])
        sum = 0
        for i in range(len(sources)):
            overlap = overlaps[i]
            print("directory:" + directory + ", iter:" +
                  str(i + 1) + "/" + str(len(sources)))
            # Save path for source & target pcd.
            source_path = 'Datasets/eth/' + directory + '/' + sources[i]
            target_path = 'Datasets/eth/' + directory + '/' + targets[i]

            # Init voxel for less num of point clouds.
            voxel_size = 0.2  # means 20cm for this dataset

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget = UR.prepare_dataset(
                voxel_size, source_path, target_path, translation_M[i], "voxel", VISUALIZATION)
            aaa = np.asarray(source.points)
            # Prepare source weight for sinkhorn with dust bin.
            source_arr = np.asarray(source_fpfh.data).T
            s = (np.ones((source_arr.shape[0]+1))
                 * (overlap.astype(float)))/source_arr.shape[0]
            s[(source_arr.shape[0])] = 1-overlap.astype(float)
            # Prepare target weight for sinkhorn with dust bin.
            target_arr = np.asarray(target_fpfh.data).T
            t = (np.ones((target_arr.shape[0]+1))
                 * (overlap.astype(float)))/target_arr.shape[0]
            t[(target_arr.shape[0])] = 1-overlap.astype(float)

            # Prepare loss matrix for sinkhorn.
            M = np.asarray(ot.dist(source_arr, target_arr))

            # Prepare dust bin for loss matrix M.
            row_to_be_added = np.zeros(((target_arr.shape[0])))
            column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
            M = np.vstack([M, row_to_be_added])
            M = np.vstack([M.T, column_to_be_added])
            M = M.T

            # Run sinkhorn with dust bin for find corr.
            sink = np.asarray(ot.sinkhorn(
                s, t, M, 100, numItermax=1200, stopThr=1e-9, verbose=False, method='sinkhorn'))

            # Take number of top corr from sinkhorn result and print result.
            corr_size = 100
            corr = np.zeros((corr_size, 2))
            for j in range(corr_size):
                max = np.unravel_index(np.argmax(sink, axis=None), sink.shape)
                corr[j][0], corr[j][1] = max[0], max[1]
                sink[max[0], :] = 0
                sink[:, max[1]] = 0

            aaa = np.asarray(source.points)
            # For sinkhorn correspondence result - run first glabal(RANSAC) and then local(ICP) regestration
            result_ransac = UR.execute_global_registration_with_corr(
                source_down, target_down, corr)
            # Execute local registration by ICP , Originals pcd and the global registration transformation result,
            # print the result and the correspondence point set .
            aaa = np.asarray(source.points)
            result_icp = UR.refine_registration_sinkhorn_ransac(
                source, target, result_ransac)
            # If the result is not bigger then the overlap else if result bigger than the overlap BUT STILL MISMATCH
            aaa = np.asarray(source.points)

            if(result_icp.fitness > overlaps[i]):
                fitness = 2 - (result_icp.fitness / overlaps[i])
            else:
                fitness = (result_icp.fitness / overlaps[i])
            results[iter_dataset].append(
                [sources[i] + " " + targets[i], fitness])
            sum += results[iter_dataset][i][1]

            print(results[iter_dataset][i][0], "fitness =", fitness)
            print("avarage score until now =",
                  sum / len(results[iter_dataset]))
            UR.draw_registration_result(
                source, target, result_icp.transformation, "ICP result")

        avg_result_datasets.append(
            [directory, sum / len(results[iter_dataset])])
        print("\navg result of dataset", directory, "is",
              avg_result_datasets[iter_dataset][1], "\n\n")
        sum_datasets += avg_result_datasets[iter_dataset][1]
        iter_dataset += 1

    total_avg = sum_datasets / len(avg_result_datasets)
    for i in range(len(avg_result_datasets)):
        print(avg_result_datasets[i][0],
              '\'s score: ', avg_result_datasets[i][1])
    print("\ntotal avarage = ", total_avg)
