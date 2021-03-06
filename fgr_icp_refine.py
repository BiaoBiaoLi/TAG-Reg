import glob
import os

import open3d as o3d
import numpy as np
import time
import copy
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt



def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    trans = np.eye(4)       #prepare matrix to return
    truth = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for  line in fr.readlines():
        line = line.strip()
        # listFromLine = line.split('\t')
        listFromLine = line.split()
        listFromLine = [float(x) for x in listFromLine]

        if(index % 5 ==0):
            index = 0
        elif(index % 5 ==1):
            trans[0, :] = np.array(listFromLine)
        elif(index % 5 ==2):
            trans[1,:] = np.array(listFromLine)
        elif(index % 5 ==3):
            trans[2,:] = np.array(listFromLine)
        elif(index % 5 ==4):
            trans[3,:] = np.array(listFromLine)
            truth.append(trans.copy())#这里不用copy的话，，，每个元素都是一样的
        index += 1
    return truth


if __name__ == '__main__':
    root_path = '/Bill/DataSet/RedWood/'
    dataset_names = ['loft', 'lobby', 'apartment','bedroom','boardroom']
    # dataset_names = ['loft']
    methods = ['ransac','fgr']

    # root_save_path = '/ransac/src2ref/refine'
    # trans_rough_all = np.load('./loft/ransac/src2ref/0_252trans_all_src2ref.npy')
    dataset_numbers = [252,199,319,219,243]
    # dataset_numbers = [44]
    for m in range(len(methods)):

        for i in  range(len(dataset_names)):
        # for i in  range(1):
            root_save_path = './'+ dataset_names[i]+'/'+ methods[m] +'/src2ref/refine'
            file_path = root_path + dataset_names[i]
            end = dataset_numbers[i]
            save_path = root_save_path
            print(file_path)
            path = glob.glob(os.path.join('./'+ dataset_names[i] +'/'+ methods[m] +'/src2ref/', '*trans_all_src2ref.npy'))

            trans_rough_all = np.load(path[0])
            groud_truth = file2matrix(file_path + '/reg_output.log')

            voxel_size = 0.05  # means 5cm for this dataset
            err_R = []
            err_T = []
            refine_results = []
            trans_all = []
            fail_list = []
            start = 0
            total_trans = np.identity(4)
            groud_truth_all = np.identity(4)
            # end = 251
            for j in range(start, end):
                print(
                    'j',j
                )
                # index_src = j + 1
                # index_ref = j

                index_src = j
                index_ref = j + 1
                current_transformation = trans_rough_all[j]
                source_show = o3d.io.read_point_cloud(file_path + "/mesh_%s.ply"%(index_src))
                target_show = o3d.io.read_point_cloud(file_path + "/mesh_%s.ply"%(index_ref))
                source = source_show.voxel_down_sample(voxel_size)
                target = target_show.voxel_down_sample(voxel_size)

                source.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
                target.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))

                result_icp = o3d.pipelines.registration.registration_icp(
                    source, target, 0.2, current_transformation,  # 后边这个距离阈值，就是大于这个距离的就去掉不计算
                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
                print(result_icp.transformation,trans_rough_all[j])
                refine_results.append(result_icp.transformation)
                total_trans = result_icp.transformation
                R = total_trans[:3,:3].reshape(3,3)
                t = total_trans[:3,3].reshape(-1,1)
                if index_src > index_ref:
                    err_R.append(np.arccos((np.trace(R.T @ groud_truth[j][:3,:3]) - 1) / 2) * 180 / np.pi )
                    err_T.append(np.linalg.norm(t - groud_truth[j][:3,3].reshape(-1,1), ord=2,axis=0))
                    trans_all.append((total_trans))
                else:
                    err_R.append( np.arccos( (np.trace(R @ groud_truth[j][:3,:3] ) - 1) / 2) * 180 / np.pi )
                    err_T.append(np.linalg.norm(-R.T @ t - groud_truth[j][:3,3].reshape(-1,1), ord=2,axis=0))
                    trans_all.append((total_trans))

                # print(total_trans[:3,:3] @ groud_truth[j][:3,:3], np.trace(total_trans[:3,:3] @ groud_truth[j][:3,:3] - np.eye(3)))
                # print(total_trans, groud_truth[j])
                # print('err_R err_T', err_R, err_T,total_trans)
            if index_src > index_ref:
            #
            #     location = str(start) + '_' + str(end)

                err_all = [err_R, err_T]
                plt.figure("ERR_R ref2src")  # 图像窗口名称
                plt.plot(err_R)
                plt.savefig(save_path + '/%s_%s_err_All_ref2src.jpg'%(start, end))
                plt.close()
                plt.figure("ERR_T ref2src")  # 图像窗口名称
                plt.plot(err_T)
                plt.savefig(save_path + '/%s_%s_trans_all_ref2src.jpg' % (start, end))
                plt.close()
                np.savetxt(save_path + '/%s_%s_fail_list_ref2src.txt'%(start, end), fail_list)
                np.save(save_path + '/%s_%s_err_All_ref2src.npy'%(start, end), err_all)
                np.savetxt(save_path + '/%s_%s_err_All_ref2src.txt' % (start, end), err_all)
                np.save(save_path + '/%s_%s_trans_all_ref2src.npy'%(start, end), trans_all)
                np.savetxt(save_path + '/%s_%s_trans_all_ref2src.txt'%(start, end), np.array(trans_all).reshape(-1,4),fmt='%0.8f')

            else:
                err_all = [err_R, err_T]
                plt.figure("ERR_R src2ref")  # 图像窗口名称refine_results
                plt.plot(err_R)
                plt.savefig(save_path +  '/%s_%serr_All_src2ref.jpg'%(start, end))
                plt.close()
                plt.figure("ERR_T src2ref")  # 图像窗口名称
                plt.plot(err_T)
                plt.savefig(save_path + '/%s_%strans_all_src2ref.jpg' % (start, end))
                plt.close()
                np.savetxt(save_path + '/%s_%s_fail_list_src2ref.txt'%(start, end), fail_list)
                np.savetxt(save_path + '/%s_%serr_All_src2ref.txt' % (start, end), err_all)
                np.save(save_path + '/%s_%serr_All_src2ref.npy'%(start, end), err_all)
                np.save(save_path + '/%s_%strans_all_src2ref.npy'%(start, end), trans_all)
                np.save(save_path + '/%s_%srefine_results_all_src2ref.npy'%(start, end), refine_results)
                np.savetxt(save_path + '/%s_%strans_all_src2ref.txt'%(start, end), np.array(trans_all).reshape(-1,4),fmt='%0.8f')
                np.savetxt(save_path + '/%s_%srefine_results_all_src2ref.txt'%(start, end), np.array(refine_results).reshape(-1,4),fmt='%0.8f')