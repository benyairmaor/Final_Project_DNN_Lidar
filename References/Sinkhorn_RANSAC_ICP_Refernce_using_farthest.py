import numpy as np
import UtilitiesReference as UR
import ot
import os
import open3d as o3d

VISUALIZATION = False
VERBOSE = True

if __name__ == '__main__':

    # Initialization parameters for model

    directories = ['apartment', 'hauptgebaude', 'wood_autumn',
                   'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs',  'plain']
    results = []
    size_dataset = []
    iter_dataset = 0

    score_per_dataset_corr_matches = 0
    score_all_datasets_corr_matches = 0
    matches_corr_matches = []
    avg_result_datasets_corr_matches = []
    problems_idx_solved_corr_matches = []
    problems_idx_unsolved_corr_matches = []

    score_per_dataset_overlap = 0
    score_all_datasets_overlap = 0
    matches_overlap = []
    avg_result_datasets_overlap = []
    problems_idx_solved_overlap = []
    problems_idx_unsolved_overlap = []

    score_all_datasets_matrix_dist_rotation = 0
    score_all_datasets_matrix_dist_translation = 0
    score_per_dataset_matrix_dist_rotation = 0
    score_per_dataset_matrix_dist_translation = 0
    matches_matrix_dist = []
    avg_result_datasets_matrix_dist = []
    problems_idx_solved_matrix_dist = []
    problems_idx_unsolved_matrix_dist = []

    score_per_dataset_RMSE = 0
    score_all_datasets_RMSE = 0
    matches_RMSE = []
    avg_result_datasets_RMSE = []
    problems_idx_solved_RMSE = []
    problems_idx_unsolved_RMSE = []

    for directory in directories:

        # Get all problem
        sources, targets, overlaps, translation_M = UR.get_data_global(
            directory, True)

        # Initialization parameters per dataset
        results.append([])
        size_dataset.append(len(sources))

        matches_corr_matches.append(0)
        score_per_dataset_corr_matches = 0
        problems_idx_solved_corr_matches.append([])
        problems_idx_unsolved_corr_matches.append([])

        matches_overlap.append(0)
        score_per_dataset_overlap = 0
        problems_idx_solved_overlap.append([])
        problems_idx_unsolved_overlap.append([])

        matches_matrix_dist.append(0)
        score_per_dataset_matrix_dist_rotation = 0
        score_per_dataset_matrix_dist_translation = 0
        problems_idx_solved_matrix_dist.append([])
        problems_idx_unsolved_matrix_dist.append([])

        matches_RMSE.append(0)
        score_per_dataset_RMSE = 0
        problems_idx_solved_RMSE.append([])
        problems_idx_unsolved_RMSE.append([])

        for i in range(len(sources)):

            if VERBOSE:
                print("\ndirectory:" + directory + ", iter:" +
                      str(i + 1) + "/" + str(len(sources)), "\n")

            overlap = overlaps[i]

            # Save path for source & target pcd.
            source_path = 'Datasets/eth/' + directory + '/' + sources[i]
            target_path = 'Datasets/eth/' + directory + '/' + targets[i]

            method, _, typeM = os.path.basename(__file__).partition('using_')
            typeM, _, _ = typeM.partition('.py')
            path_saveImg = 'images/eth/' + \
                directory + '/' + method + "_" + typeM + "_" + str(i)

            # Init voxel for less num of point clouds.
            voxel_size = 0.1  # means 5cm for this dataset ?

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget = UR.prepare_dataset(
                voxel_size, source_path, target_path, translation_M[i], "fartest_point", VISUALIZATION, pathSaveImg=path_saveImg, directoryName=directory)

            source_arr = np.asarray(source_fpfh.data).T
            s = (np.ones((source_arr.shape[0]+1))
                 * (overlap.astype(float)))/source_arr.shape[0]
            s[(source_arr.shape[0])] = 1-overlap.astype(float)
            # Prepare target weight for sinkhorn with dust bin.
            target_arr = np.asarray(target_fpfh.data).T
            t = (np.ones((target_arr.shape[0]+1))
                 * (overlap.astype(float)))/target_arr.shape[0]
            t[(target_arr.shape[0])] = 1-overlap.astype(float)

            if VERBOSE:
                # Print weights and shapes of a and b weight vectors.
                print("source FPFH shape: ", source_arr.shape,
                      "\ntarget FPFH shape: ", target_arr.shape)
                print("source weight shape: ", s.shape,
                      "\ntarget weight shape: ", t.shape)

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

            if VERBOSE:
                # Print loss matrix shape
                print("Loss matrix m shape : ", M.shape)

            # Run sinkhorn with dust bin for find corr.
            sink = np.asarray(ot.sinkhorn_unbalanced(s, t, M, reg=100, reg_m=100,
                              numItermax=12000, stopThr=1e-16, verbose=False, method='sinkhorn_stabilized'))

            # Take number of top corr from sinkhorn result, take also the corr weights and print corr result.
            corr_size = int(0.15 * overlap *
                            np.minimum(M.shape[0]-1, M.shape[1]-1))
            corr = np.zeros((corr_size, 2))
            corr_weights = np.zeros((corr_size, 1))
            j = 0
            sink[M.shape[0]-1, :] = 0
            sink[:, M.shape[1]-1] = 0
            while j < corr_size:
                max = np.unravel_index(np.argmax(sink, axis=None), sink.shape)
                corr[j][0], corr[j][1] = max[0], max[1]
                # Save corr weights.
                corr_weights[j] = sink[max[0], max[1]]  # Pn
                sink[max[0], :] = 0
                sink[:, max[1]] = 0
                j = j+1

            # Build numpy array for original points
            source_arr = np.asarray(source_down.points)
            target_arr = np.asarray(target_down.points)

            # Take only the relevant indexes (without dust bin)
            corr_values_source = source_arr[corr[:, 0].astype(int), :]  # Xn
            corr_values_target = target_arr[corr[:, 1].astype(int), :]  # Yn

            pcdS = o3d.geometry.PointCloud()
            pcdS.points = o3d.utility.Vector3dVector(corr_values_source)
            pcdT = o3d.geometry.PointCloud()
            pcdT.points = o3d.utility.Vector3dVector(corr_values_target)
            if VISUALIZATION:
                UR.draw_registration_result(
                    pcdS, pcdT, np.identity(4), "Corr set")
            UR.savePCDS(pcdS, pcdT, "Corr_set_Sinkhorn",
                        path_saveImg, np.identity(4), directory)

            # For sinkhorn correspondence result - run first glabal(RANSAC) and then local(ICP) regestration
            result_ransac = UR.execute_global_registration_with_corr(
                source_down, target_down, corr)

            # Build numpy array for original points
            source_arr = np.asarray(source_down.points)
            target_arr = np.asarray(target_down.points)

            # Take only the relevant indexes (without dust bin)
            # Xn
            corr_values_source = source_arr[np.asarray(
                result_ransac.correspondence_set)[:, 0], :]
            # Yn
            corr_values_target = target_arr[np.asarray(
                result_ransac.correspondence_set)[:, 1], :]

            pcdS = o3d.geometry.PointCloud()
            pcdS.points = o3d.utility.Vector3dVector(corr_values_source)
            pcdT = o3d.geometry.PointCloud()
            pcdT.points = o3d.utility.Vector3dVector(corr_values_target)
            if VISUALIZATION:
                UR.draw_registration_result(
                    pcdS, pcdT, np.identity(4), "Corr set")
            UR.savePCDS(pcdS, pcdT, "Corr_set_RANSAC",
                        path_saveImg, np.identity(4), directory)

            # Execute local registration by ICP , Originals pcd and the global registration transformation result,
            # print the result and the correspondence point set .
            result_icp = UR.refine_registration_sinkhorn_ransac(
                source, target, result_ransac)
            res = result_icp.transformation
            # If the result is not bigger then the overlap else if result bigger than the overlap BUT STILL MISMATCH

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

            # Check how many problems solved with score above 70%
            if fitness > 0.7:
                matches_corr_matches[iter_dataset] += 1
                problems_idx_solved_corr_matches[iter_dataset].append(i)
            else:
                problems_idx_unsolved_corr_matches[iter_dataset].append(i)

            # 2. Calculate the overlap beteen the PCDs
            overlap_score = 0

            if(result_icp.fitness > overlaps[i]):
                overlap_score = 2 - (result_icp.fitness / overlaps[i])
            else:
                overlap_score = (result_icp.fitness / overlaps[i])

            # Check how many problems solved with score above 70%
            if overlap_score > 0.7:
                matches_overlap[iter_dataset] += 1
                problems_idx_solved_overlap[iter_dataset].append(i)
            else:
                problems_idx_unsolved_overlap[iter_dataset].append(i)

            RMSE_score = result_icp.inlier_rmse
            # Check how many problems solved with score above 70%
            if RMSE_score < 0.3:
                matches_RMSE[iter_dataset] += 1
                problems_idx_solved_RMSE[iter_dataset].append(i)
            else:
                problems_idx_unsolved_RMSE[iter_dataset].append(i)

            # 3. Compute the distanse between the resulp ICP translation matrix and the inverse of the problem matrix
            rotaition_score = np.linalg.norm(
                res[:3, :3] - np.linalg.inv(translation_M[i])[:3, :3])
            translation_score = np.linalg.norm(
                res[:3, 3:] - np.linalg.inv(translation_M[i])[:3, 3:])

            # Check how many problems solved with score above 70%
            if rotaition_score < 1 and translation_score < 1.5:
                matches_matrix_dist[iter_dataset] += 1
                problems_idx_solved_matrix_dist[iter_dataset].append(i)
            else:
                problems_idx_unsolved_matrix_dist[iter_dataset].append(i)

            results[iter_dataset].append(
                [sources[i] + " " + targets[i], fitness, overlap_score, rotaition_score, translation_score, RMSE_score])

            # Calculate the total per problem per approch
            score_per_dataset_corr_matches += results[iter_dataset][i][1]
            score_per_dataset_overlap += results[iter_dataset][i][2]
            score_per_dataset_matrix_dist_rotation += results[iter_dataset][i][3]
            score_per_dataset_matrix_dist_translation += results[iter_dataset][i][4]
            score_per_dataset_RMSE += results[iter_dataset][i][5]

            if VERBOSE:
                print(results[iter_dataset][i][0], "fitness =", fitness, "overlap =", overlap_score,
                      "matrix distance = (rotation)", rotaition_score, "(translation)", translation_score)
                print("avarage fitness score until now =",
                      score_per_dataset_corr_matches / len(results[iter_dataset]))
                print("avarage overlap score until now =",
                      score_per_dataset_overlap / len(results[iter_dataset]))
                print("avarage matrix distance score until now for rotation =",
                      score_per_dataset_matrix_dist_rotation / len(results[iter_dataset]))
                print("avarage matrix distance score until now for translation =",
                      score_per_dataset_matrix_dist_translation / len(results[iter_dataset]))
                print("avarage RMSE score until now =",
                      score_per_dataset_RMSE / len(results[iter_dataset]))

            if VISUALIZATION:
                UR.draw_registration_result(
                    source, target, res, "Sinkhorn_RANSAC_ICP result")
            UR.savePCDS(source, target, "Sinkhorn_RANSAC_ICP_result",
                        path_saveImg, res, directory)

        avg_result_datasets_corr_matches.append(
            [directory, score_per_dataset_corr_matches / len(results[iter_dataset])])
        avg_result_datasets_overlap.append(
            [directory, score_per_dataset_overlap / len(results[iter_dataset])])
        avg_result_datasets_matrix_dist.append(
            [directory, score_per_dataset_matrix_dist_rotation / len(results[iter_dataset]), score_per_dataset_matrix_dist_translation / len(results[iter_dataset])])
        avg_result_datasets_RMSE.append(
            [directory, score_per_dataset_RMSE / len(results[iter_dataset])])

        if VERBOSE:
            print("\n(fitness) avg result of dataset", directory, "is",
                  avg_result_datasets_corr_matches[iter_dataset][1])
            print("(overlap) avg result of dataset", directory, "is",
                  avg_result_datasets_overlap[iter_dataset][1])
            print("(matrix distance) avg result of dataset", directory, "is",
                  avg_result_datasets_matrix_dist[iter_dataset][1], avg_result_datasets_matrix_dist[iter_dataset][2])
            print("(RMSE) avg result of dataset", directory, "is",
                  avg_result_datasets_RMSE[iter_dataset][1])

        # Sum the score per dataset per approch
        score_all_datasets_corr_matches += avg_result_datasets_corr_matches[iter_dataset][1]
        score_all_datasets_overlap += avg_result_datasets_overlap[iter_dataset][1]
        score_all_datasets_matrix_dist_rotation += avg_result_datasets_matrix_dist[iter_dataset][1]
        score_all_datasets_matrix_dist_translation += avg_result_datasets_matrix_dist[iter_dataset][2]
        score_all_datasets_RMSE += avg_result_datasets_RMSE[iter_dataset][1]

        iter_dataset += 1

    # Calculate the total score per dataset per approch
    total_avg_corr_matches = score_all_datasets_corr_matches / \
        len(avg_result_datasets_corr_matches)
    total_avg_overlap = score_all_datasets_overlap / \
        len(avg_result_datasets_overlap)
    total_avg_matrix_dist_rotation = score_all_datasets_matrix_dist_rotation / \
        len(avg_result_datasets_matrix_dist)
    total_avg_matrix_dist_translation = score_all_datasets_matrix_dist_translation / \
        len(avg_result_datasets_matrix_dist)
    total_avg_RMSE = score_all_datasets_RMSE / \
        len(avg_result_datasets_RMSE)

    if VERBOSE:
        print()
        for i in range(len(avg_result_datasets_corr_matches)):
            print(avg_result_datasets_corr_matches[i][0], '\'s fitness score: ', avg_result_datasets_corr_matches[i]
                  [1], 'with ', matches_corr_matches[i], 'problems solved with score over 70% from ', size_dataset[i])
            print("Problem indexes solved:",
                  problems_idx_solved_corr_matches[i])
            print("Problem indexes unsolved:",
                  problems_idx_unsolved_corr_matches[i])
            print(avg_result_datasets_overlap[i][0], '\'s overlap score: ', avg_result_datasets_overlap[i]
                  [1], 'with ', matches_overlap[i], 'problems solved with score over 70% from ', size_dataset[i])
            print("Problem indexes solved:", problems_idx_solved_overlap[i])
            print("Problem indexes unsolved:",
                  problems_idx_unsolved_overlap[i])
            print(avg_result_datasets_matrix_dist[i][0], '\'s matrix distance score: ', avg_result_datasets_matrix_dist[i]
                  [1], avg_result_datasets_matrix_dist[i][2], 'with ', matches_matrix_dist[i], 'problems solved with score under 1 from ', size_dataset[i])
            print("Problem indexes solved:",
                  problems_idx_solved_matrix_dist[i])
            print("Problem indexes unsolved:",
                  problems_idx_unsolved_matrix_dist[i])
            print(avg_result_datasets_RMSE[i][0], '\'s RMSE score: ', avg_result_datasets_RMSE[i]
                  [1], 'with ', matches_RMSE[i], 'problems solved with score over 70% from ', size_dataset[i])
            print("Problem indexes solved:", problems_idx_solved_RMSE[i])
            print("Problem indexes unsolved:",
                  problems_idx_unsolved_RMSE[i])
            print()
        print("total avarage (fitness) = ", total_avg_corr_matches)
        print("total avarage (overlap) = ", total_avg_overlap)
        print("total avarage (matrix distance) = ",
              total_avg_matrix_dist_rotation, total_avg_matrix_dist_translation)
        print("total avarage (RMSE) = ", total_avg_RMSE)
