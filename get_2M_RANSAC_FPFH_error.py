import open3d as o3d
import numpy as np
import time
import copy
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    target_temp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # print('source_fpfh', source_fpfh.num, target_fpfh.num)
    #     print('source_fpfh',source_fpfh,np.asarray(source_fpfh.data))
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 500))
    return result


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    trans = np.eye(4)  # prepare matrix to return
    truth = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        # listFromLine = line.split('\t')
        listFromLine = line.split()
        listFromLine = [float(x) for x in listFromLine]

        if (index % 5 == 0):
            index = 0
        elif (index % 5 == 1):
            trans[0, :] = np.array(listFromLine)
        elif (index % 5 == 2):
            trans[1, :] = np.array(listFromLine)
        elif (index % 5 == 3):
            trans[2, :] = np.array(listFromLine)
        elif (index % 5 == 4):
            trans[3, :] = np.array(listFromLine)
            truth.append(trans.copy())  # 这里不用copy的话，，，每个元素都是一样的
        index += 1
    return truth


if __name__ == '__main__':
    root_path = '/Bill/DataSet/RedWood/'
    dataset_names = ['loft', 'lobby', 'apartment', 'bedroom', 'boardroom']
    root_save_path = '/ransac_2M/src2ref'
    dataset_numbers = [252, 199, 319, 219, 243]
    for i in range(len(dataset_names)):
        # for i in  range(1):
        file_path = root_path + dataset_names[i]
        end = dataset_numbers[i]
        save_path = dataset_names[i] + root_save_path

        print(file_path)
        groud_truth = file2matrix(file_path + '/reg_output.log')

        voxel_size = 0.05  # means 5cm for this dataset
        err_R = []
        err_T = []
        trans_all = []
        fail_list = []
        start = 0
        # end = 251
        for j in range(start, end):
            print(
                'j', j
            )
            # index_src = j + 1
            # index_ref = j

            index_src = j
            index_ref = j + 1

            source_show = o3d.io.read_point_cloud(file_path + "/mesh_%s.ply" % (index_src))
            target_show = o3d.io.read_point_cloud(file_path + "/mesh_%s.ply" % (index_ref))
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_show,
                                                                                                 target_show,
                                                                                                 voxel_size)
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
            print(result_ransac.transformation)
            total_trans = result_ransac.transformation

            R = total_trans[:3, :3].reshape(3, 3)
            t = total_trans[:3, 3].reshape(-1, 1)
            if index_src > index_ref:
                err_R.append(np.arccos((np.trace(R.T @ groud_truth[j][:3, :3]) - 1) / 2) * 180 / np.pi)
                err_T.append(np.linalg.norm(t - groud_truth[j][:3, 3].reshape(-1, 1), ord=2, axis=0))
                trans_all.append((total_trans))
            else:
                err_R.append(np.arccos((np.trace(R @ groud_truth[j][:3, :3]) - 1) / 2) * 180 / np.pi)
                err_T.append(np.linalg.norm(-R.T @ t - groud_truth[j][:3, 3].reshape(-1, 1), ord=2, axis=0))
                trans_all.append((total_trans))

            # print(total_trans[:3,:3] @ groud_truth[j][:3,:3], np.trace(total_trans[:3,:3] @ groud_truth[j][:3,:3] - np.eye(3)))
            # print(total_trans, groud_truth[j])
            print('err_R err_T', err_R[j - start], err_T[j - start], total_trans)
        if index_src > index_ref:
            #
            #     location = str(start) + '_' + str(end)

            err_all = [err_R, err_T]
            plt.figure("ERR_R ref2src")  # 图像窗口名称
            plt.plot(err_R)
            plt.savefig(save_path + '/%s_%s_err_All_ref2src.jpg' % (start, end))
            # plt.show()
            plt.close()
            plt.figure("ERR_T ref2src")  # 图像窗口名称
            plt.plot(err_T)
            plt.savefig(save_path + '/%s_%s_trans_all_ref2src.jpg' % (start, end))
            # plt.show()
            plt.close()
            np.savetxt(save_path + '/%s_%s_fail_list_ref2src.txt' % (start, end), fail_list)
            np.save(save_path + '/%s_%s_err_All_ref2src.npy' % (start, end), err_all)
            np.savetxt(save_path + '/%s_%s_err_All_ref2src.txt' % (start, end), err_all)
            np.save(save_path + '/%s_%s_trans_all_ref2src.npy' % (start, end), trans_all)
            np.savetxt(save_path + '/%s_%s_trans_all_ref2src.txt' % (start, end), np.array(trans_all).reshape(-1, 4),
                       fmt='%0.8f')

        else:
            err_all = [err_R, err_T]
            plt.figure("ERR_R src2ref")  # 图像窗口名称
            plt.plot(err_R)
            plt.savefig(save_path + '/%s_%serr_All_src2ref.jpg' % (start, end))
            # plt.show()
            plt.close()
            plt.figure("ERR_T src2ref")  # 图像窗口名称
            plt.plot(err_T)
            plt.savefig(save_path + '/%s_%strans_all_src2ref.jpg' % (start, end))
            # plt.show()
            plt.close()
            np.savetxt(save_path + '/%s_%s_fail_list_src2ref.txt' % (start, end), fail_list)
            np.savetxt(save_path + '/%s_%serr_All_src2ref.txt' % (start, end), err_all)
            np.save(save_path + '/%s_%serr_All_src2ref.npy' % (start, end), err_all)
            np.save(save_path + '/%s_%strans_all_src2ref.npy' % (start, end), trans_all)
            np.savetxt(save_path + '/%s_%strans_all_src2ref.txt' % (start, end), np.array(trans_all).reshape(-1, 4),
                       fmt='%0.8f')
