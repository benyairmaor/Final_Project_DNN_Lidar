import numpy as np
import open3d as o3d
import copy
import ot
import pandas as pd
import UtilitiesReference as UR

if __name__ == '__main__':

    directories = ['apartment', 'hauptgebaude', 'wood_autumn', 'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs', 'plain']
    results = []
    matches = []
    size_dataset = []
    avg_result_datasets = []
    iter_dataset = 0
    sum = 0
    sum_datasets = 0
    for directory in directories:
        sources, targets, overlaps, translation_M = UR.get_data_global(directory)
        results.append([])
        matches.append(0)
        size_dataset.append(len(sources))
        sum = 0
        for i in range(len(sources) - 1, len(sources)):
            i = 0
            print("directory:" + directory + ", iter:" + str(i + 1) + "/" + str(len(sources)))
            # Save path for source & target pcd.
            source_path = 'Datasets/eth/' + directory + '/' + sources[i]
            target_path = 'Datasets/eth/' + directory + '/' + targets[i]

            # Init voxel for less num of point clouds.
            voxel_size = 1  # means 20cm for this dataset

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_down_c, target_down_c,  source_fpfh, target_fpfh, M_result, listSource, listTarget = UR.prepare_dataset(voxel_size, source_path, target_path, translation_M[i])
            # Prepare source weight for sinkhorn with dust bin.
            source_arr = np.asarray(source_fpfh.data).T
            s = (np.ones((source_arr.shape[0]+1))*(2/3))/source_arr.shape[0]
            s[(source_arr.shape[0])] = 1/3
            # Prepare target weight for sinkhorn with dust bin.
            target_arr = np.asarray(target_fpfh.data).T
            t = (np.ones((target_arr.shape[0]+1))*(2/3))/target_arr.shape[0]
            t[(target_arr.shape[0])] = 1/3

            # Prepare loss matrix for sinkhorn.
            M = np.asarray(ot.dist(source_arr, target_arr))

            # Prepare dust bin for loss matrix M.
            row_to_be_added = np.zeros(((target_arr.shape[0])))
            column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
            M = np.vstack([M, row_to_be_added])
            M = np.vstack([M.T, column_to_be_added])
            M = M.T

            # Run sinkhorn with dust bin for find corr.
            sink = np.asarray(ot.sinkhorn(s, t, M, 100, numItermax=1200, stopThr=1e-9, verbose=False, method='sinkhorn'))

            # Take number of top corr from sinkhorn result and print result.
            corr_size = 150
            corr = np.zeros((corr_size, 2))
            for j in range(corr_size):
                max = np.unravel_index(np.argmax(sink, axis=None), sink.shape)
                corr[j][0], corr[j][1] = max[0], max[1]
                sink[max[0], :] = 0
                sink[:, max[1]] = 0
            
            # For sinkhorn correspondence result - run first glabal(RANSAC) and then local(ICP) regestration
            result_ransac = UR.execute_global_registration_with_corr(source_down, target_down, corr)
            UR.draw_registration_result(source, target, result_ransac.transformation, "ransac result")
            # Execute local registration by ICP , Originals pcd and the global registration transformation result,
            # print the result and the correspondence point set .
            result_icp = UR.refine_registration_sinkhorn_ransac(source, target, result_ransac)
            
            # Calculate the score by compare the correspondnce before and after the tarsformtion.
            source_down_c.transform(result_icp.transformation)
            fitness_ = 0.
            M_check = np.asarray(ot.dist(np.asarray(source_down_c.points), np.asarray(target_down_c.points)))
            for idx in range(len(listSource)):
                    if M_check[listSource[idx], listTarget[idx]] <= 0.1001:
                        fitness_ += 1
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.2002:
                        fitness_ += 0.8
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.3003:
                        fitness_ += 0.6
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.4004:
                        fitness_ += 0.4
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.4004:
                        fitness_ += 0.2
            
            fitness = fitness_ / np.sum(M_result)

            matrix_score = np.linalg.norm(result_icp.transformation - np.linalg.inv(translation_M))
            
            overlap_score = 0
            if(result_icp.fitness > overlaps[i]):
                overlap_score = 2 - (result_icp.fitness / overlaps[i])
            else:
                overlap_score = (result_icp.fitness / overlaps[i])
            
            results[iter_dataset].append([sources[i] + " " + targets[i], fitness, matrix_score, overlap_score])
            
            if fitness > 0.7:
                matches[iter_dataset] += 1
                sum += results[iter_dataset][i][1]

            print(results[iter_dataset][i][0], "fitness =", fitness, "matrix_score = ", matrix_score, "overlap_score = ", overlap_score)
            print("avarage score until now =",sum / len(results[iter_dataset]))
            UR.draw_registration_result(source, target, result_icp.transformation, "ICP result")
            source.transform(result_icp.transformation)
            i = len(sources) - 1

        avg_result_datasets.append([directory, sum / len(results[iter_dataset])])
        print("\navg result of dataset", directory, "is",avg_result_datasets[iter_dataset][1], "\n\n")
        sum_datasets += avg_result_datasets[iter_dataset][1]
        iter_dataset += 1

    total_avg = sum_datasets / len(avg_result_datasets)
    print()
    for i in range(len(avg_result_datasets)):
        print(avg_result_datasets[i][0], '\'s score: ', avg_result_datasets[i][1], 'with ', matches[i], 'problems solved from ', size_dataset[i])
    print()
    print("\ntotal avarage = ", total_avg)
