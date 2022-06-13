import numpy as np
import UtilitiesReference as UR
import ot
import matplotlib.pyplot as plt

VISUALIZATION = False
VERBOSE = True

if __name__ == '__main__':

    # Initialization parameters for model

    directories = ['hauptgebaude', 'wood_autumn',
                   'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs',  'plain']
    # , 'hauptgebaude', 'wood_autumn',
    #    'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs',  'plain']
    idx_s = [3, 3, 7, 5, 2, 9, 8, 6]
    idx_f = [4, 1, -1, 1, 0, 4, -1, 2]
    farthests_size = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.06])
    voxel_size = 0.1
    scores_fitness = np.ones((2, len(directories), len(farthests_size))) * -1
    scores_overlap = np.ones((2, len(directories), len(farthests_size))) * -1
    scores_matrix_distance_rotation = np.ones(
        (2, len(directories), len(farthests_size))) * -1
    scores_matrix_distance_translation = np.ones(
        (2, len(directories), len(farthests_size))) * -1
    size_dataset = []
    iter_dataset = 0

    score_per_dataset_corr_matches = 0
    score_per_dataset_overlap = 0
    score_all_datasets_matrix_dist_rotation = 0

    for directory in directories:

        # Get all problem
        sources, targets, overlaps, translation_M = UR.get_data_global(
            directory, True)

        # Initialization parameters per dataset
        size_dataset.append(len(sources))
        score_per_dataset_corr_matches = 0
        score_per_dataset_overlap = 0
        score_per_dataset_matrix_dist_rotation = 0
        score_per_dataset_matrix_dist_translation = 0

        for j in range(2):
            for farthest_size in range(len(farthests_size)):
                if j == 0:
                    i = idx_s[iter_dataset]
                else:
                    i = idx_f[iter_dataset]
                if idx_s == -1 or idx_f == -1:
                    break

                # Save path for source & target pcd.
                source_path = 'Datasets/eth/' + directory + '/' + sources[i]
                target_path = 'Datasets/eth/' + directory + '/' + targets[i]

                # Prepare data set by compute FPFH.
                source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget = UR.prepare_dataset(
                    voxel_size, source_path, target_path, translation_M[i], "fartest_point", VISUALIZATION, farthests_size[farthest_size])

                # Execute global registration by RANSAC and FPFH , print the result and the correspondence point set .
                result_ransac = UR.execute_global_registration(
                    source_down, target_down, source_fpfh, target_fpfh)

                # Execute local registration by ICP , Originals pcd and the global registration transformation result,
                # print the result and the correspondence point set .
                result_icp = UR.refine_registration(
                    source, target, result_ransac)

                # Calculate the score by 3 diffenerte approaches
                # 1. Compare the correspondnce before and after the tarsformtion. (fitness) as far as target point from source the socre in decreases.
                source_down_c.transform(result_icp.transformation)
                fitness_ = 0.
                M_check = np.asarray(
                    ot.dist(np.asarray(source_down_c.points), np.asarray(target_down_c.points)))

                for idx in range(len(listSource)):
                    if M_check[listSource[idx], listTarget[idx]] <= 0.1001:
                        fitness_ += 1
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.2002:
                        fitness_ += 0.8
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.3003:
                        fitness_ += 0.6
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.4004:
                        fitness_ += 0.4
                    elif M_check[listSource[idx], listTarget[idx]] <= 0.5005:
                        fitness_ += 0.2

                fitness = fitness_ / np.sum(M_result)

                # 2. Calculate the overlap beteen the PCDs
                overlap_score = 0

                if(result_icp.fitness > overlaps[i]):
                    overlap_score = 2 - (result_icp.fitness / overlaps[i])
                else:
                    overlap_score = (result_icp.fitness / overlaps[i])

                # 3. Compute the distanse between the resulp ICP translation matrix and the inverse of the problem matrix
                rotaition_score = np.linalg.norm(
                    result_icp.transformation[:3, :3] - np.linalg.inv(translation_M[i])[:3, :3])
                translation_score = np.linalg.norm(
                    result_icp.transformation[:3, 3:] - np.linalg.inv(translation_M[i])[:3, 3:])

                if VISUALIZATION:
                    UR.draw_registration_result(
                        source, target, result_icp.transformation, "ICP result")

                scores_fitness[j][iter_dataset][farthest_size] = fitness
                scores_overlap[j][iter_dataset][farthest_size] = overlap_score
                scores_matrix_distance_rotation[j][iter_dataset][farthest_size] = rotaition_score
                scores_matrix_distance_translation[j][iter_dataset][farthest_size] = translation_score
        iter_dataset += 1

    for y in range(scores_fitness.shape[1]):
        for x in range(scores_fitness.shape[0]):
            plt.plot(farthests_size, scores_fitness[x][y], color='green')
            plt.xlabel('farthests size')
            plt.ylabel('scores (fitness)')

            # displaying the title
            if x == 0:
                plt.title(directories[y] +
                          " Correct solution - fitness per farthest")
            else:
                plt.title(directories[y] +
                          " Uncorrect solution - fitness per farthest")

            # plt.show()
            plt.savefig(directories[y]+'_'+x+'_fitness.png')

            plt.plot(farthests_size, scores_overlap[x][y], color='green')
            plt.xlabel('farthests size')
            plt.ylabel('scores (overlap)')

            if x == 0:
                plt.title(directories[y] + " Correct - overlap per farthest")
            else:
                plt.title(directories[y] + " Uncorrect - overlap per farthest")

            # plt.show()
            plt.savefig(directories[y]+'_'+x+'_overlap.png')

            plt.plot(
                farthests_size, scores_matrix_distance_rotation[x][y], color='green')
            plt.xlabel('farthests size')
            plt.ylabel('scores (matrix distance rotation)')

            if x == 0:
                plt.title(
                    directories[y] + " Correct - matrix distance rotation per farthest")
            else:
                plt.title(
                    directories[y] + " Uncorrect - matrix distance rotation per farthest")

            # plt.show()
            plt.savefig(directories[y]+'_'+x+'_matrix_distanceR.png')

            plt.plot(
                farthests_size, scores_matrix_distance_translation[x][y], color='green')
            plt.xlabel('farthests size')
            plt.ylabel('scores (matrix distance translation)')

            if x == 0:
                plt.title(
                    directories[y] + " Correct - matrix distance translation per farthest")
            else:
                plt.title(
                    directories[y] + " Uncorrect - matrix distance translation per farthest")

            # plt.show()
            plt.savefig(directories[y]+'_'+x+'_matrix_distanceT.png')
