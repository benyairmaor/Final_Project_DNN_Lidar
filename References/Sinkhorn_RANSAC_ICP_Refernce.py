import numpy as np
import open3d as o3d
import copy
import ot
import pandas as pd


# Method get the data from global file for POC
def get_data_global_POC(directory):
    headers = ['id', 'source', 'target', 'overlap', 't1', 't2',
               't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
    read_file = pd.read_csv('eth/' + directory + '_global_POC.txt', sep=" ", header=0, names=headers)
    read_file.to_csv('eth/' + directory + '_global_POC.csv', sep=',')
    read_file = pd.DataFrame(read_file, columns=headers)
    M = np.zeros((len(read_file), 4, 4))
    for row in range(len(read_file)):
        for i in range(1, 13):
            idx_row = int((i - 1) / 4)
            idx_col = (i - 1) % 4
            M[row, idx_row, idx_col] = read_file['t' + str(i)][row]
        M[row, 3, :] = [0, 0, 0, 1]
    return read_file['source'], read_file['target'], read_file['overlap'], M


# For draw source & target point cloud.
def draw_registration_result(source, target, transformation, title):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp], title, 1280, 720, True)


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 5
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    radius_feature = voxel_size * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
    return pcd_down, pcd_fpfh


# For loading the point clouds : return -
# (original source , original target , voxel down source , voxel down target , FPFH source , FPFH target).
def prepare_dataset(voxel_size, source_path, target_path, trans_init):
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path))
    transformation = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]
    draw_registration_result(source, target, transformation, "Target Matching")
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# Run global regestration by RANSAC
def execute_global_registration_with_corr(source_down, target_down, corr):
    distance_threshold = 0.1001
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        True)
    ransac_n = 3
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold)
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        1000000, 0.9999)
    seed = 1
    corr_size = 0.1001
    corr = o3d.utility.Vector2iVector(corr)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_down, target_down, corr, corr_size,
        estimation_method,
        ransac_n,
        checkers,
        criteria,
        seed)

    draw_registration_result(source_down, target_down,
                             result.transformation, "RANSAC result")
    return result

# Run local regestration by icp with transformation result from global regestretion such as RANSAC.


def refine_registration(source, target, result_ransac):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1200))
    draw_registration_result(
        source, target, result.transformation, "ICP result")
    return result


if __name__ == '__main__':

    directories = ['apartment', 'hauptgebaude', 'wood_autumn', 'gazebo_summer', 'gazebo_winter',
                   'wood_summer', 'stairs', 'plain']
    results = []
    avg_result_datasets = []
    iter_dataset = 0
    sum = 0
    sum_datasets = 0
    for directory in directories:
        sources, targets, overlaps, translation_M = get_data_global_POC(
            directory)
        results.append([])
        sum = 0
        for i in range(len(sources)):
            print("directory:" + directory + ", iter:" +
                  str(i + 1) + "/" + str(len(sources)))
            # Save path for source & target pcd.
            source_path = 'eth/' + directory + '/' + sources[i]
            target_path = 'eth/' + directory + '/' + targets[i]

            # Init voxel for less num of point clouds.
            voxel_size = 0.2  # means 20cm for this dataset

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
                voxel_size, source_path, target_path, translation_M[i])

            # Prepare source weight for sinkhorn with dust bin.
            source_arr = np.asarray(source_fpfh.data).T
            s = (np.ones((source_arr.shape[0]+1))*(2/3))/source_arr.shape[0]
            s[(source_arr.shape[0])] = 1/3
            # Prepare target weight for sinkhorn with dust bin.
            target_arr = np.asarray(target_fpfh.data).T
            t = (np.ones((target_arr.shape[0]+1))*(2/3))/target_arr.shape[0]
            t[(target_arr.shape[0])] = 1/3

            # Print weights and shapes of a and b weight vectors.
            # print("source FPFH shape: ", source_arr.shape,
            #       "\ntarget FPFH shape: ", target_arr.shape)
            # print("source weight shape: ", s.shape,
            #       "\ntarget weight shape: ", t.shape)

            # Prepare loss matrix for sinkhorn.
            M = np.asarray(ot.dist(source_arr, target_arr))

            # Prepare dust bin for loss matrix M.
            row_to_be_added = np.zeros(((target_arr.shape[0])))
            column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
            M = np.vstack([M, row_to_be_added])
            M = np.vstack([M.T, column_to_be_added])
            M = M.T
            # Print loss matrix and shape
            # print("Loss matrix m shape : ", M.shape)

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
            # print("Correspondence set shape: ", corr.shape)

            # For sinkhorn correspondence result - run first glabal(RANSAC) and then local(ICP) regestration
            result_ransac = execute_global_registration_with_corr(
                source_down, target_down, corr)
            # print(result_ransac)
            # print("Ransac correspondence_set: ", np.asarray(
            #     result_ransac.correspondence_set), "\n")

            # Execute local registration by ICP , Originals pcd and the global registration transformation result,
            # print the result and the correspondence point set .
            result_icp = refine_registration(source, target, result_ransac)
            # print(result_icp)
            # print("Icp correspondence_set: ", np.asarray(
            #     result_icp.correspondence_set), "\n")
            # If the result is not bigger then the overlap else if result bigger than the overlap BUT STILL MISMATCH
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
            # draw_registration_result(
            #     source, target, result_icp.transformation, "ICP result")
        avg_result_datasets.append(
            [directory, sum / len(results[iter_dataset])])
        print("\navg result of dataset", directory, "is",
              avg_result_datasets[iter_dataset][1], "\n")
        sum_datasets += avg_result_datasets[iter_dataset][1]
        iter_dataset += 1
    total_avg = sum_datasets / len(avg_result_datasets)
    for i in range(len(avg_result_datasets)):
        # print(directories[i], '\'s score: ', avg_result_datasets[i])
        print(avg_result_datasets[i][0],
              '\'s score: ', avg_result_datasets[i][1])
    print("total avarage = ", total_avg)
