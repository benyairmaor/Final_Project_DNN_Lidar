import numpy as np
import open3d as o3d
import copy
import ot


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


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
    print("The original overlap: ", o)
    # draw_registration_result(source, target, np.identity(4), "Original")
    trans_init = np.asarray([[-0.5754197841861329, 0.817372954385317,
                              -0.028169583003715, 11.778369303008173],
                             [-0.7611987839242382, -0.5478349625282469,
                              -0.34706377682485917, 14.264281414042465],
                             [-0.2991128270727379, -0.17826471123330384,
                              0.9374183747982869, 1.6731349336747363],
                             [0., 0., 0., 1.]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4), "Original problem")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    print("\n", "source size: ", source, "\n", "target size: ", target,
          "\n", "source_down size: ", source_down, "\n", "target_down size: ",  target_down,
          "\n", "source_fpfh size: ", source_fpfh, "\n", "target_fpfh size: ", target_fpfh, "\n")

    return source, target, source_down, target_down, source_fpfh, target_fpfh


if __name__ == '__main__':
    # Save path for source & target pcd.
    source_path = "eth//hauptgebaude//PointCloud26.pcd"
    target_path = "eth//hauptgebaude//PointCloud27.pcd"

    # Init voxel for less num of point clouds.
    voxel_size = 0.2  # means voxel_size-cm for this dataset.

    # Prepare data set by compute FPFH.
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size, source_path, target_path)

    # Prepare source weight for sinkhorn with dust bin.
    source_arr = np.asarray(source_fpfh.data).T
    source_down_arr = np.asarray(source_down.points)
    s = (np.ones((source_arr.shape[0]+1))*0.7)/source_arr.shape[0]
    s[(source_arr.shape[0])] = 0.3
    # Prepare target weight for sinkhorn with dust bin.
    target_arr = np.asarray(target_fpfh.data).T
    target_down_arr = np.asarray(target_down.points)
    t = (np.ones((target_arr.shape[0]+1))*0.7)/target_arr.shape[0]
    t[(target_arr.shape[0])] = 0.3

    # Print weights and shapes of a and b weight vectors.
    print("source weight: ", s, "\ntarget weight: ", t)
    print(source_arr.shape, target_arr.shape)

    # Prepare loss matrix for sinkhorn.
    M = ot.dist(source_arr, target_arr)
    # Prepare dust bin for loss matrix M.
    row_to_be_added = np.zeros(((target_arr.shape[0])))
    column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
    M = np.vstack([M, row_to_be_added])
    M = np.vstack([M.T, column_to_be_added])
    M = M.T
    # Print loss matrix and shape
    print("Loss matrix m : ", M)
    print(M.shape)

    # Run sinkhorn for find top correspondence.
    sink = np.asarray(ot.sinkhorn(
        s, t, M, 1000, numItermax=10000, verbose=True))

    # Take number of top corr from sinkhorn result and print result.
    corr_size = 100
    corr = np.zeros((corr_size, 2))
    for i in range(corr_size):
        max = np.unravel_index(np.argmax(sink, axis=None), sink.shape)
        corr[i][0], corr[i][1] = max[0], max[1]
        sink[max[0], :] = 0
        sink[:, max[1]] = 0
    print("Correspondence indexes set shape: ", corr.shape,
          "\nCorrespondence indexes set: ", corr)

    print("check: ", source_down_arr.shape, corr.shape)

    # Take only the relevant indexes (without dust bin)
    corr_values_source = source_down_arr[corr[2:, 0].astype(int), :]
    corr_values_target = target_down_arr[corr[2:, 1].astype(int), :]
    corr_values = np.column_stack((corr_values_source, corr_values_target))
    print("Correspondence vlaues set shape: ", corr_values.shape,
          "\nCorrespondence vlaues set: ", corr_values)
    corr_tensor = o3d.core.Tensor(corr_values)
    u, s, v_transpose = o3d.core.svd(corr_tensor)
    print("SVD result - u shape: ", u.shape)
    print("SVD result - s shape: ", s.shape)
    print("SVD result - v_transpose shape: ", v_transpose.shape)
