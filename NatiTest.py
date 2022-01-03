import sys
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
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
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
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
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

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        estimation_method,
        ransac_n,
        checkers,
        criteria,
        seed)

    o = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.1001, result.transformation)
    print("RANSAC FULLY DATA RESULT: ", o)

    # draw_registration_result(source_down, target_down,
    #                          result.transformation, "RANSAC result")
    return result

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
    corr_size = corr.shape[0]
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

    # draw_registration_result(source_down, target_down,
    #                          result.transformation, "RANSAC result")
    return result

# Run local regestration by icp with transformation result from global regestretion such as RANSAC.


def refine_registration(source, target, result_ransac):
    distance_threshold = 0.1001
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # draw_registration_result(
    #     source, target, result.transformation, "ICP result")
    return result


if __name__ == '__main__':
    # Save path for source & target pcd.
    source_path = "eth//hauptgebaude//PointCloud26.pcd"
    target_path = "eth//hauptgebaude//PointCloud27.pcd"

    # Init voxel for less num of point clouds.
    voxel_size = 0.05  # means 5cm for this dataset ?

    # Prepare data set by compute FPFH.
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size, source_path, target_path)

    # Execute global registration by RANSAC and FPFH , print the result and the correspondence point set .
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    print("Ransac correspondence_set: ", np.asarray(
        result_ransac.correspondence_set), "\n")

    # Execute local registration by ICP , Originals pcd and the global registration transformation result,
    # print the result and the correspondence point set .
    result_icp = refine_registration(
        source, target, result_ransac)
    print(result_icp)
    print("Icp correspondence_set: ", np.asarray(
        result_icp.correspondence_set), "\n")

    # Sample of sinkhorn.
    source_arr = np.asarray(source_fpfh.data).T
    s = np.ones((source_arr.shape[0]))/source_arr.shape[0]
    plt.subplot(121)
    plt.hist(s)

    target_arr = np.asarray(target_fpfh.data).T
    t = np.ones((target_arr.shape[0]))/target_arr.shape[0]
    plt.subplot(122)
    plt.hist(t)
    # plt.show()
    print("s: ", s, "t: ", t)
    print(source_arr.shape, target_arr.shape)

    M = ot.dist(source_arr, target_arr)
    print(M.shape)
    sink = np.asarray(ot.sinkhorn(
        s, t, M, 100, numItermax=10000, verbose=True))

    corr_size = 20
    corr = np.zeros((corr_size, 2))
    for i in range(corr_size):
        max = np.unravel_index(np.argmax(sink, axis=None), sink.shape)
        corr[i][0], corr[i][1] = max[0], max[1]
        sink[max[0], :] = 0
        sink[:, max[1]] = 0

    print(corr)

    result_ransac = execute_global_registration_with_corr(
        source_down, target_down, corr)
    print(result_ransac)
    print("Ransac correspondence_set: ", np.asarray(
        result_ransac.correspondence_set), "\n")

    # Execute local registration by ICP , Originals pcd and the global registration transformation result,
    # print the result and the correspondence point set .
    result_icp = refine_registration(
        source, target, result_ransac)
    print(result_icp)
    print("Icp correspondence_set: ", np.asarray(
        result_icp.correspondence_set), "\n")
