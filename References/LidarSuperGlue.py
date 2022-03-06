import math
import numpy as np
import open3d as o3d
import copy
import ot


def findCorr(source, target, distanceThreshold):
    list = []
    threshold = []
    tragetCopy = copy.deepcopy(target)
    sourceCopy = copy.deepcopy(source)
    for pointS_Idx in range(len(sourceCopy)):
        for pointT_Idx in range(len(tragetCopy)):
            if (pointS_Idx, pointT_Idx) not in list:
                dist = np.linalg.norm(
                    source[pointS_Idx]-tragetCopy[pointT_Idx])
                if dist <= distanceThreshold:
                    print(dist, " ", source[pointS_Idx],
                          " ", target[pointT_Idx])
                    list.append((pointS_Idx, pointT_Idx))
                    threshold.append(dist)
                # tragetCopy = np.delete(tragetCopy, pointT_Idx)
                break
    return list


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp], width=1280, height=720)


# For pre prossecing the point cloud - make voxel and compute FPFH.
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 5
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=250))
    radius_feature = voxel_size * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=600))
    return pcd_down, pcd_fpfh


# For loading the point clouds : return -
# (original source , original target , voxel down source , voxel down target , FPFH source , FPFH target).
def prepare_dataset(voxel_size, source_path, target_path):
    source = copy.deepcopy(o3d.io.read_point_cloud(source_path))
    target = copy.deepcopy(o3d.io.read_point_cloud(target_path))
    draw_registration_result(source, target, np.identity(4))
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

    # Calculate KeyPoints
    print("TEST KEYPOINTS")
    key_source = o3d.geometry.keypoint.compute_iss_keypoints(source)
    key_target = o3d.geometry.keypoint.compute_iss_keypoints(target)
    print("num of key_source", key_source)
    print("num of key_target", key_target)
    draw_registration_result(key_source, key_target, np.identity(4))

    # Find correspondence
    s_keyPointArr = np.asarray(key_source.points)
    t_keyPointArr = np.asarray(key_target.points)
    print(len(s_keyPointArr), len(t_keyPointArr),
          s_keyPointArr.shape, t_keyPointArr.shape)
    corrList = findCorr(s_keyPointArr, t_keyPointArr, 0.1001)
    print(len(corrList), corrList)

    # Test corr
    # print(len(s_keyPointArr[corrList[0], :]))
    pcdS = o3d.geometry.PointCloud()
    pcdS.points = o3d.utility.Vector3dVector(s_keyPointArr[corrList[0], :])
    pcdT = o3d.geometry.PointCloud()
    pcdT.points = o3d.utility.Vector3dVector(t_keyPointArr[corrList[1], :])
    draw_registration_result(pcdS, pcdT, np.identity(4))

    # Make transform (the problem)
    trans_init = np.asarray([[-0.5754197841861329, 0.817372954385317,
                              -0.028169583003715, 11.778369303008173],
                             [-0.7611987839242382, -0.5478349625282469,
                              -0.34706377682485917, 14.264281414042465],
                             [-0.2991128270727379, -0.17826471123330384,
                              0.9374183747982869, 1.6731349336747363],
                             [0., 0., 0., 1.]])
    key_source.transform(trans_init)
    draw_registration_result(key_source, key_target, np.identity(4))

    # indexArrS = []
    # indexArrT = []
    # keyArrS = np.asarray(key_source.points)
    # keyArrT = np.asarray(key_target.points)
    # ArrS = np.asarray(source.points)
    # ArrT = np.asarray(target.points)
    # for i in range(len(keyArrS)):
    #     print(i)
    #     for j in range(len(ArrS)):
    #         if keyArrS[i][0] == ArrS[j][0] and keyArrS[i][1] == ArrS[j][1] and keyArrS[i][2] == ArrS[j][2]:
    #             indexArrS.append(j)
    #             break
    # for i in range(len(keyArrT)):
    #     print(i)
    #     for j in range(len(ArrT)):
    #         if keyArrT[i][0] == ArrT[j][0] and keyArrT[i][1] == ArrT[j][1] and keyArrT[i][2] == ArrT[j][2]:
    #             indexArrT.append(j)
    #             break

    # print("indexArrS: ", len(indexArrS))
    # print("indexArrT: ", len(indexArrT))
