import numpy as np
import open3d as o3d
import copy
import ot
import UtilitiesReference as UR

if __name__ == '__main__':

    directories = ['apartment', 'hauptgebaude', 'wood_autumn', 'gazebo_summer', 'gazebo_winter',
                   'wood_summer', 'stairs', 'plain']
    results = []
    avg_result_datasets = []
    iter_dataset = 0
    sum = 0
    sum_datasets = 0
    for directory in directories:
        sources, targets, overlaps, translation_M = UR.get_data_global(
            directory)
        results.append([])
        sum = 0
        for i in range(len(sources)):
            print("directory:" + directory + ", iter:" +
                  str(i + 1) + "/" + str(len(sources)))
            # Save path for source & target pcd.
            source_path = 'Datasets/eth/' + directory + '/' + sources[i]
            target_path = 'Datasets/eth/' + directory + '/' + targets[i]

            # Init voxel for less num of point clouds.
            voxel_size = 0.2  # means voxel_size-cm for this dataset.

            # Prepare data set by compute FPFH.
            source, target, source_down, target_down, source_fpfh, target_fpfh = UR.prepare_dataset(
                voxel_size, source_path, target_path, translation_M[i])

            # Prepare source weight for sinkhorn with dust bin.
            source_arr = np.asarray(source_fpfh.data).T
            s = (np.ones((source_arr.shape[0]+1))*(0.6))/source_arr.shape[0]
            s[(source_arr.shape[0])] = 0.4
            # Prepare target weight for sinkhorn with dust bin.
            target_arr = np.asarray(target_fpfh.data).T
            t = (np.ones((target_arr.shape[0]+1))*(0.6))/target_arr.shape[0]
            t[(target_arr.shape[0])] = 0.4

            # Print weights and shapes of a and b weight vectors.
            print("source FPFH shape: ", source_arr.shape,
                  "\ntarget FPFH shape: ", target_arr.shape)
            print("source weight shape: ", s.shape,
                  "\ntarget weight shape: ", t.shape)

            # Prepare loss matrix for sinkhorn.
            M = np.asarray(ot.dist(source_arr, target_arr))

            # Prepare dust bin for loss matrix M.
            row_to_be_added = np.zeros(((target_arr.shape[0])))
            column_to_be_added = np.zeros(((source_arr.shape[0]+1)))
            M = np.vstack([M, row_to_be_added])
            M = np.vstack([M.T, column_to_be_added])
            M = M.T
            # Print loss matrix shape
            print("Loss matrix m shape : ", M.shape)

            # Run sinkhorn with dust bin for find corr.
            sink = np.asarray(ot.sinkhorn(
                s, t, M, 100, numItermax=1200, stopThr=1e-9, verbose=False, method='sinkhorn'))

            # Take number of top corr from sinkhorn result, take also the corr weights and print corr result.
            corr_size = 500
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
            print("Correspondence set index values: ", corr)

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
            UR.draw_registration_result(pcdS, pcdT, np.identity(4), "debug")

            # Norm to sum equal to one for corr weights.
            # corr_weights = (corr_weights / np.sum(corr_weights))  # Pn norm

            # Calc the mean of source and target point/FPFH with respect to points weight.
            source_mean = np.sum(
                corr_values_source*corr_weights, axis=0)/np.sum(corr_weights)  # X0
            target_mean = np.sum(
                corr_values_target*corr_weights, axis=0)/np.sum(corr_weights)  # Y0

            # Calc the mean-reduced coordinate for Y and X
            corr_values_source = corr_values_source-source_mean  # An
            corr_values_target = corr_values_target-target_mean  # Bn

            print(corr_values_source.shape, corr_values_target.shape,
                  corr_weights.shape, source_mean.shape, target_mean.shape)

            # Compute the cross-covariance matrix H
            H = np.zeros((3, 3))
            for k in range(corr_size):
                H = H + np.outer(corr_values_source[k, :],
                                 corr_values_target[k, :]) * corr_weights[k]

            # Print for debug
            print("corr_values_source: ", corr_values_source.shape, "\ncorr_values_target: ",
                  corr_values_target.shape, "\ncorr_weights: ", corr_weights, "\ncorr_weights sum: ", np.sum(
                      corr_weights),
                  "\nsource mean shape: ", source_mean.shape, "\ntarget mean shape: ", target_mean.shape,
                  "\nsource mean: ", source_mean, "\ntarget mean: ", target_mean,
                  "\ncovariance matrix shape: ", H.shape, "\ncovariance matrix:\n", H)

            # Calc SVD to cross-covariance matrix H.
            corr_tensor = o3d.core.Tensor(H)
            u, s, v_transpose = o3d.core.svd(corr_tensor)
            u, s, v_transpose = u.numpy(), s.numpy(), v_transpose.numpy()
            print("SVD result - u:\n", u)
            print("SVD result - s:\n", s)
            print("SVD result - v_transpose:\n", v_transpose)

            # Calc R and t from SVD result u and v transpose
            R = (v_transpose.T) @ (u.T)
            t = target_mean - R@source_mean

            # Calc the transform matrix from R and t
            res = np.vstack([R.T, t])
            res = res.T
            res = np.vstack([res, np.array([0, 0, 0, 1])])
            print("R:\n", R, "\nt:\n", t, "\ntransform res:\n",
                  res, "\ninvers of original:\n")

            # Check the transform matrix result
            UR.draw_registration_result(source, target, res, "Sinkhorn SVD")
