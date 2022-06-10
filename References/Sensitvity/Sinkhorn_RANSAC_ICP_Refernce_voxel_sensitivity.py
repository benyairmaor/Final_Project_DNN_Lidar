import numpy as np
import UtilitiesReference as UR
import ot
import matplotlib.pyplot as plt
import open3d as o3d

VISUALIZATION = False
VERBOSE = True

if __name__ == '__main__':

    # Initialization parameters for model

    directories = ['apartment', 'hauptgebaude', 'wood_autumn',
                   'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs',  'plain']
    idx_s = [3, 3, 7, 5, 2, 9, 8, 6]
    idx_f = [4, 1, -1, 1, 0, 4, -1, 2]
    voxels_size = np.array([0.1, 0.2, 0.5, 1, 1.5, 2, 5])
    scores_fitness = np.ones((2, len(directories), len(voxels_size))) * -1
    scores_overlap = np.ones((2, len(directories), len(voxels_size))) * -1
    scores_matrix_distance_rotation = np.ones(
        (2, len(directories), len(voxels_size))) * -1
    scores_matrix_distance_translation = np.ones(
        (2, len(directories), len(voxels_size))) * -1
    size_dataset = []
    iter_dataset = 0

    score_per_dataset_corr_matches = 0
    score_per_dataset_overlap = 0
    score_all_datasets_matrix_dist_rotation = 0

    for directory in directories:
        print(directory)
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
            for voxel_size in range(len(voxels_size)):
                print(j, voxel_size, '/', len(voxels_size))
                if j == 0:
                    i = idx_s[iter_dataset]
                else:
                    i = idx_f[iter_dataset]
                if i == -1:
                    break

                overlap = overlaps[i]
                # Save path for source & target pcd.
                source_path = 'Datasets/eth/' + directory + '/' + sources[i]
                target_path = 'Datasets/eth/' + directory + '/' + targets[i]

                # Prepare data set by compute FPFH.
                source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget = UR.prepare_dataset(
                    voxels_size[voxel_size], source_path, target_path, translation_M[i], "voxel", VISUALIZATION)

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
                # M = np.einsum('dn,dm->nm', source_arr.T, target_arr.T)
                # normalizeEinsumResult(M, source_arr, target_arr)

                # Prepare dust bin for loss matrix M.
                row_to_be_added = np.zeros(((target_arr.shape[0])))
                column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
                M = np.vstack([M, row_to_be_added])
                M = np.vstack([M.T, column_to_be_added])
                M = M.T

                # Run sinkhorn with dust bin for find corr.
                sink = np.asarray(ot.sinkhorn_unbalanced(s, t, M, reg=100, reg_m=100,
                                                         numItermax=12000, stopThr=1e-16, verbose=False, method='sinkhorn_stabilized'))

                # Take number of top corr from sinkhorn result, take also the corr weights and print corr result.
                corr_size = int(0.15 * overlap *
                                np.minimum(M.shape[0]-1, M.shape[1]-1))
                corr = np.zeros((corr_size, 2))
                corr_weights = np.zeros((corr_size, 1))
                j_ = 0
                sink[M.shape[0]-1, :] = 0
                sink[:, M.shape[1]-1] = 0
                while j_ < corr_size:
                    max = np.unravel_index(
                        np.argmax(sink, axis=None), sink.shape)
                    corr[j_][0], corr[j_][1] = max[0], max[1]
                    # Save corr weights.
                    corr_weights[j_] = sink[max[0], max[1]]  # Pn
                    sink[max[0], :] = 0
                    sink[:, max[1]] = 0
                    j_ = j_+1

                # Build numpy array for original points
                source_arr = np.asarray(source_down.points)
                target_arr = np.asarray(target_down.points)

                # Take only the relevant indexes (without dust bin)
                corr_values_source = source_arr[corr[:, 0].astype(
                    int), :]  # Xn
                corr_values_target = target_arr[corr[:, 1].astype(
                    int), :]  # Yn

                pcdS = o3d.geometry.PointCloud()
                pcdS.points = o3d.utility.Vector3dVector(corr_values_source)
                pcdT = o3d.geometry.PointCloud()
                pcdT.points = o3d.utility.Vector3dVector(corr_values_target)
                if VISUALIZATION:
                    UR.draw_registration_result(
                        pcdS, pcdT, np.identity(4), "Corr set")

                # For sinkhorn correspondence result - run first glabal(RANSAC) and then local(ICP) regestration
                result_ransac = UR.execute_global_registration_with_corr(
                    source_down, target_down, corr)
                # Execute local registration by ICP , Originals pcd and the global registration transformation result,
                # print the result and the correspondence point set .
                result_icp = UR.refine_registration_sinkhorn_ransac(
                    source, target, result_ransac)
                res = result_icp.transformation

                # Calculate the score by 3 diffenerte approaches
                # 1. Compare the correspondnce before and after the tarsformtion. (fitness) as far as target point from source the socre in decreases.
                source_down_c.transform(res)
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
                    res[:3, :3] - np.linalg.inv(translation_M[i])[:3, :3])
                translation_score = np.linalg.norm(
                    res[:3, 3:] - np.linalg.inv(translation_M[i])[:3, 3:])

                if VISUALIZATION:
                    UR.draw_registration_result(
                        source, target, res, "Sinkhorn_RANSAC_ICP result")

                scores_fitness[j][iter_dataset][voxel_size] = fitness
                scores_overlap[j][iter_dataset][voxel_size] = overlap_score
                scores_matrix_distance_rotation[j][iter_dataset][voxel_size] = rotaition_score
                scores_matrix_distance_translation[j][iter_dataset][voxel_size] = translation_score
        iter_dataset += 1

    for y in range(scores_fitness.shape[1]):
        for x in range(scores_fitness.shape[0]):
            plt.plot(voxels_size, scores_fitness[x][y], color='green')
            plt.xlabel('voxels size')
            plt.ylabel('scores (fitness)')

            # displaying the title
            if x == 0:
                plt.title(directories[y] +
                          " Correct solution - fitness per voxel")
            else:
                plt.title(directories[y] +
                          " Uncorrect solution - fitness per voxel")

            plt.show()

            plt.plot(voxels_size, scores_overlap[x][y], color='green')
            plt.xlabel('voxels size')
            plt.ylabel('scores (overlap)')

            if x == 0:
                plt.title(directories[y] + " Correct - overlap per voxel")
            else:
                plt.title(directories[y] + " Uncorrect - overlap per voxel")

            plt.show()

            plt.plot(
                voxels_size, scores_matrix_distance_rotation[x][y], color='green')
            plt.xlabel('voxels size')
            plt.ylabel('scores (matrix distance rotation)')

            if x == 0:
                plt.title(
                    directories[y] + " Correct - matrix distance rotation per voxel")
            else:
                plt.title(
                    directories[y] + " Uncorrect - matrix distance rotation per voxel")

            plt.show()

            plt.plot(
                voxels_size, scores_matrix_distance_translation[x][y], color='green')
            plt.xlabel('voxels size')
            plt.ylabel('scores (matrix distance translation)')

            if x == 0:
                plt.title(
                    directories[y] + " Correct - matrix distance translation per voxel")
            else:
                plt.title(
                    directories[y] + " Uncorrect - matrix distance translation per voxel")

            plt.show()
