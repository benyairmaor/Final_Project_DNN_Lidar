import numpy as np
import open3d as o3d
import copy
import pandas as pd
import UtilitiesReference as UR
import ot

if __name__ == '__main__':
    directories = ['apartment', 'hauptgebaude', 'wood_autumn',
                   'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs',  'plain']
    results = []
    matches = []
    size_dataset = []
    avg_result_datasets = []
    iter_dataset = 0
    sum = 0
    sum_datasets = 0
    for directory in directories:
        sources, targets, overlaps, translation_M = UR.get_data_global(
            directory)
        results.append([])
        matches.append(0)
        size_dataset.append(len(sources))
        sum = 0
        for i in range(len(sources)):
            print("directory:" + directory + ", iter:" +
                  str(i + 1) + "/" + str(len(sources)))
            # Save path for source & target pcd.
            source_path = 'eth/' + directory + '/' + sources[i]
            target_path = 'eth/' + directory + '/' + targets[i]

            # Init voxel for less num of point clouds.
            voxel_size = 0.1  # means 5cm for this dataset ?

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget = UR.prepare_dataset(
                voxel_size, source_path, target_path, translation_M[i])

            # Execute global registration by RANSAC and FPFH , print the result and the correspondence point set .
            result_ransac = UR.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh)

            # Execute local registration by ICP , Originals pcd and the global registration transformation result,
            # print the result and the correspondence point set .
            result_icp = UR.refine_registration(source, target, result_ransac)

            # Calculate the score by compare the correspondnce before and after the tarsformtion.
            source_down_c.transform(result_icp.transformation)
            fitness_ = 0.
            M_check = np.asarray(
                ot.dist(np.asarray(source_down_c.points), np.asarray(target_down_c.points)))
            for idx in range(len(listSource)):
                if M_check[listSource[idx], listTarget[idx]] <= 0.1001:
                    fitness_ += 1

            fitness = fitness_ / np.sum(M_result)

            results[iter_dataset].append(
                [sources[i] + " " + targets[i], fitness])
            if fitness > 0.7:
                matches[iter_dataset] += 1
            sum += results[iter_dataset][i][1]

            print(results[iter_dataset][i][0], "fitness =", fitness)
            print("avarage score until now =",
                  sum / len(results[iter_dataset]))
            UR.draw_registration_result(
                source, target, result_icp.transformation, "ICP result")

        avg_result_datasets.append(
            [directory, sum / len(results[iter_dataset])])
        print("avg result of dataset", directory, "is",
              avg_result_datasets[iter_dataset][1])
        sum_datasets += avg_result_datasets[iter_dataset][1]
        iter_dataset += 1

    total_avg = sum_datasets / len(avg_result_datasets)
    print()
    for i in range(len(avg_result_datasets)):
        print(avg_result_datasets[i][0], '\'s score: ', avg_result_datasets[i]
              [1], 'with ', matches[i], 'problems solved from ', size_dataset[i])
    print()
    print("total avarage = ", total_avg)
