import copy
import numpy as np
import UtilitiesReference as UR
import ot
import matplotlib.pyplot as plt
import open3d as o3d

VISUALIZATION = True
VERBOSE = True


def testSVD(corr, percentOfCorrTofuck):
    percentOfCorrTofuck = percentOfCorrTofuck/2
    halfSize = int(corr.shape[0]*percentOfCorrTofuck)
    for i in range(halfSize):
        j = corr.shape[0]-1-i
        tmp = corr[i][1]
        corr[i][1] = corr[j][1]
        corr[j][1] = tmp
    return corr


if __name__ == '__main__':

    # Initialization parameters for model

    directories = ['hauptgebaude']
    idx_s = [3]
    voxels_size = 0.3
    scores_fitness = []
    scores_matrix_distance_rotation = []
    scores_matrix_distance_translation = []
    percentOfCorrTofuckList = [0.0, 0.1, 0.2,
                               0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    directory = directories[0]
    print(directory)
    # Get all problem
    sources, targets, overlaps, translation_M = UR.get_data_global(
        directory, True)
    # Initialization parameters per dataset
    score_per_dataset_corr_matches = 0
    score_per_dataset_overlap = 0
    score_per_dataset_matrix_dist_rotation = 0
    score_per_dataset_matrix_dist_translation = 0
    source_path = 'Datasets/eth/' + directory + '/' + sources[3]
    target_path = 'Datasets/eth/' + directory + '/' + targets[3]
    target_path = source_path
    # Prepare data set by compute FPFH.
    source, target, source_down, target_down, source_down_c, target_down_c, source_fpfh, target_fpfh, M_result, listSource, listTarget = UR.prepare_dataset(
        voxels_size, source_path, target_path, translation_M[3], "voxel", VISUALIZATION)

    source_arr = np.asarray(source_fpfh.data).T
    s = (np.ones((source_arr.shape[0]+1)))/source_arr.shape[0]
    s[(source_arr.shape[0])] = 0
    # Prepare target weight for sinkhorn with dust bin.
    target_arr = np.asarray(target_fpfh.data).T
    t = np.ones((target_arr.shape[0]+1))/target_arr.shape[0]
    t[(target_arr.shape[0])] = 0
    # Prepare loss matrix for sinkhorn.
    M = np.asarray(ot.dist(source_arr, target_arr))
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
    corr_size = int(0.5*np.minimum(M.shape[0]-1, M.shape[1]-1))
    corrS = np.zeros((corr_size, 2))
    corr_weights = np.zeros((corr_size, 1))
    j_ = 0
    sink[M.shape[0]-1, :] = 0
    sink[:, M.shape[1]-1] = 0
    while j_ < corr_size:
        max = np.unravel_index(
            np.argmax(sink, axis=None), sink.shape)
        corrS[j_][0], corrS[j_][1] = max[0], max[1]
        # Save corr weights.
        corr_weights[j_] = sink[max[0], max[1]]  # Pn
        sink[max[0], :] = 0
        sink[:, max[1]] = 0
        j_ = j_+1
    print(corrS)
    for percent in percentOfCorrTofuckList:
        corr = testSVD(copy.deepcopy(corrS), percent)
        # Build numpy array for original points
        result_ransac = UR.execute_global_registration_with_corr(
            source_down, target_down, corr)
        res = result_ransac.transformation

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

        # 3. Compute the distanse between the resulp ICP translation matrix and the inverse of the problem matrix
        rotaition_score = np.linalg.norm(
            res[:3, :3] - np.linalg.inv(translation_M[3])[:3, :3])
        translation_score = np.linalg.norm(
            res[:3, 3:] - np.linalg.inv(translation_M[3])[:3, 3:])

        if VISUALIZATION:
            UR.draw_registration_result(
                source, target, res, "Sinkhorn SVD result")

        scores_fitness.append(fitness)
        scores_matrix_distance_rotation.append(rotaition_score)
        scores_matrix_distance_translation.append(translation_score)
        print(scores_fitness, "\n", scores_matrix_distance_rotation,
              "\n", scores_matrix_distance_translation, "\n")

    plt.plot(percentOfCorrTofuckList,
             scores_matrix_distance_rotation, color='green')
    plt.xlabel('Percent Of correspondences to ignore')
    plt.ylabel('Matrix distance rotation score')
    plt.title("Matrix distance rotation")
    plt.savefig('RANSAC_Test_Matrix_distance_rotation.png')
    plt.close("all")

    plt.plot(percentOfCorrTofuckList,
             scores_matrix_distance_translation, color='green')
    plt.xlabel("Percent Of correspondences to ignore")
    plt.ylabel('Matrix distance translation score')
    plt.title("Matrix distance translation")
    plt.savefig('RANSAC_Test_Matrix_distance_translation.png')
    plt.close("all")
