import numpy as np
import open3d as o3d
import copy
import pandas as pd

#Method get the data from global file for POC
def get_data_global_POC(directory):
    headers = ['id', 'source', 'target', 'overlap', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
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
    o3d.visualization.draw_geometries([source_temp, target_temp], title, 1280, 720, True)


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 3
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    radius_feature = voxel_size * 7
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150))
    return pcd_down, pcd_fpfh


# For loading the point clouds : return -
# (original source , original target , voxel down source , voxel down target , FPFH source , FPFH target).
def prepare_dataset(voxel_size, source_path, target_path, trans_init):
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path))
    transformation = [[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1]]
    draw_registration_result(source, target, transformation, "Target Matching")
    source.transform(trans_init)
    draw_registration_result(source, target, transformation, "Problem")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# Run global regestration by RANSAC
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
    distance_threshold = 0.1001
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
    ransac_n = 3
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
    seed = 1

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold, estimation_method, ransac_n, checkers, criteria, seed)
    return result


# Run local regestration by icp with transformation result from global regestretion such as RANSAC.
def refine_registration(source, target, result_ransac):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 1000))
    return result


if __name__ == '__main__':

    directories = ['apartment', 'hauptgebaude', 'wood_autumn', 'gazebo_summer', 'gazebo_winter', 'wood_summer', 'stairs',  'plain']
    results = []
    avg_result_datasets = []
    iter_dataset = 0
    sum = 0
    sum_datasets = 0
    for directory in directories:
        sources, targets, overlaps, translation_M = get_data_global_POC(directory)
        results.append([])
        sum = 0
        for i in range(len(sources)):
            print("directory:" + directory + ", iter:" + str(i + 1) + "/" + str(len(sources)))
            # Save path for source & target pcd.
            source_path = 'eth/' + directory + '/' + sources[i]
            target_path = 'eth/' + directory + '/' + targets[i]

            # Init voxel for less num of point clouds.
            voxel_size = 0.1  # means 5cm for this dataset ?

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source_path, target_path, translation_M[i])

            # Execute global registration by RANSAC and FPFH , print the result and the correspondence point set .
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)

            # Execute local registration by ICP , Originals pcd and the global registration transformation result,
            # print the result and the correspondence point set .
            result_icp = refine_registration(source, target, result_ransac)
            
            # If the result is not bigger then the overlap else if result bigger than the overlap BUT STILL MISMATCH
            if(result_icp.fitness > overlaps[i]):
                fitness = 2 - (result_icp.fitness / overlaps[i])
            else: 
                fitness = (result_icp.fitness / overlaps[i])
            results[iter_dataset].append([sources[i] + " " + targets[i], fitness])
            sum += results[iter_dataset][i][1]
            print(results[iter_dataset][i][0], "fitness =", fitness)
            print("avarage score until now =", sum / len(results[iter_dataset]))
            draw_registration_result(source, target, result_icp.transformation, "ICP result")
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
