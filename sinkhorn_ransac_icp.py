import numpy as np
import open3d as o3d
import copy
import ot
import matplotlib.pyplot as plt

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
    radius_normal = voxel_size * 4
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=60))
    radius_feature = voxel_size * 8
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=120))
    return pcd_down, pcd_fpfh


# For loading the point clouds : return -
# (original source , original target , voxel down source , voxel down target , FPFH source , FPFH target).
def prepare_dataset(voxel_size, source_path, target_path):
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path))

    o = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.1001)
    print(o)
    # draw_registration_result(
    #     source, target, np.identity(4), "Original")
    trans_init = np.asarray([[-0.5754197841861329, 0.817372954385317, -0.028169583003715, 11.778369303008173],
                             [-0.7611987839242382, -0.5478349625282469,
                              -0.34706377682485917, 14.264281414042465],
                             [-0.2991128270727379, -0.17826471123330384,
                              0.9374183747982869, 1.6731349336747363],
                             [0., 0., 0., 1.]])
    source.transform(trans_init)
    # draw_registration_result(
    #     source, target, np.identity(4), "Original problem")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    print("\n", "source size: ", source, "\n", "target size: ", target,
          "\n", "source_down size: ", source_down, "\n", "target_down size: ",  target_down,
          "\n", "source_fpfh size: ", source_fpfh, "\n", "target_fpfh size: ", target_fpfh, "\n")

    return source, target, source_down, target_down, source_fpfh, target_fpfh


# Run global regestration by RANSAC
def execute_global_registration_with_corr(source_down, target_down, corr):
    distance_threshold = 0.1001
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ransac_n = 3
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold)
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        100000, 0.999)
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

    o = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.1001, result.transformation)
    print("RANSAC FULLY DATA RESULT: ", o)

    draw_registration_result(source_down, target_down,
                             result.transformation, "RANSAC result")
    return result

# Run local regestration by icp with transformation result from global regestretion such as RANSAC.


def refine_registration(source, target, result_ransac):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(
        source, target, result.transformation, "ICP result")
    return result


if __name__ == '__main__':
    # Save path for source & target pcd.
    source_path = "eth//gazebo_winter//PointCloud12.pcd"
    target_path = "eth//gazebo_winter//PointCloud13.pcd"

    # Init voxel for less num of point clouds.
    voxel_size = 0.2  # means 20cm for this dataset

    # Prepare data set by compute FPFH.
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size, source_path, target_path)

    # Prepare source weight for sinkhorn with dust bin.
    source_arr = np.asarray(source_fpfh.data).T
    s = (np.ones((source_arr.shape[0]+1))*0.7)/source_arr.shape[0]
    s[(source_arr.shape[0])] = 0.3
    # Prepare target weight for sinkhorn with dust bin.
    target_arr = np.asarray(target_fpfh.data).T
    t = (np.ones((target_arr.shape[0]+1))*0.7)/target_arr.shape[0]
    t[(target_arr.shape[0])] = 0.3

    # Print weights and shapes of a and b weight vectors.
    print("source FPFH shape: ", source_arr.shape,
          "\ntarget FPFH shape: ", target_arr.shape)
    print("source weight shape: ", s.shape, "\ntarget weight shape: ", t.shape)

    # Prepare loss matrix for sinkhorn.
    M = np.asarray(ot.dist(source_arr, target_arr))

    # Prepare dust bin for loss matrix M.
    row_to_be_added = np.zeros(((target_arr.shape[0])))
    column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
    M = np.vstack([M, row_to_be_added])
    M = np.vstack([M.T, column_to_be_added])
    M = M.T
    # Print loss matrix and shape
    print("Loss matrix m shape : ", M.shape)

    # Run sinkhorn with dust bin for find corr.
    sink = np.asarray(ot.sinkhorn(s, t, M, 100, numItermax=1000,
                      stopThr=1e-12, verbose=True, method='sinkhorn'))

    # Take number of top corr from sinkhorn result and print result.
    corr_size = 100
    corr = np.zeros((corr_size, 2))
    for i in range(corr_size):
        max = np.unravel_index(np.argmax(sink, axis=None), sink.shape)
        corr[i][0], corr[i][1] = max[0], max[1]
        sink[max[0], :] = 0
        sink[:, max[1]] = 0
    print("Correspondence set shape: ", corr.shape)

    # For sinkhorn correspondence result - run first glabal(RANSAC) and then local(ICP) regestration
    result_ransac = execute_global_registration_with_corr(
        source_down, target_down, corr)
    print(result_ransac)
    print("Ransac correspondence_set: ", np.asarray(
        result_ransac.correspondence_set), "\n")

    # Execute local registration by ICP , Originals pcd and the global registration transformation result,
    # print the result and the correspondence point set .
    result_icp = refine_registration(source, target, result_ransac)
    print(result_icp)
    print("Icp correspondence_set: ", np.asarray(
        result_icp.correspondence_set), "\n")
