import time

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import h5py
import math
import sklearn
import copy
from sklearn.neighbors import KDTree
from PIL import Image
import matplotlib.pyplot as plt


def show_point_cloud(src_, src_corr_, ref_, ref_corr_):
    src = src_.copy()
    src_corr = src_corr_.copy()
    ref = ref_.copy()
    ref_corr = ref_corr_.copy()
    ref[:,1] = ref[:,1] + 2.5
    ref_corr[:,1] = ref_corr[:,1] + 2.5
    src_pcd = o3d.geometry.PointCloud()
    src_corr_pcd = o3d.geometry.PointCloud()

    ref_pcd = o3d.geometry.PointCloud()
    ref_corr_pcd = o3d.geometry.PointCloud()

    src_pcd.points = o3d.utility.Vector3dVector(src)
    ref_pcd.points = o3d.utility.Vector3dVector(ref)
    src_corr_pcd.points = o3d.utility.Vector3dVector(src_corr)
    ref_corr_pcd.points = o3d.utility.Vector3dVector(ref_corr )

    ref_pcd.paint_uniform_color([1, 0, 0.651])  # 蓝色
    # src_corr_pcd.paint_uniform_color([1, 0.706, 0])  # 黄色
    src_pcd.paint_uniform_color([0, 0.651, 0.929])  # 红色

    line_size = src_corr.shape[0]
    line_src = np.arange(0, 2 * line_size, 2)  # 这个代表所有偶数
    rand_idxs = np.random.choice(line_size, math.ceil(line_size / 3), replace=False)
    #     print('line_src',line_src)
    line_src = line_src[rand_idxs].reshape(rand_idxs.shape[0], 1)
    #     print('line_src',line_src)
    line_ref = line_src + 1
    #     print('line_ref',line_ref)
    lines = np.concatenate([line_ref, line_src], -1).reshape(-1, 2)
    #     print('lines',lines)

    colors = [[1, 0, 0]]
    # triangle_points=np.concatenate([data['points_ref'][1, :, :3].detach().cpu().numpy()+1,data['points_src'][1, :, :3].detach().cpu().numpy()],-1)
    triangle_points = np.concatenate([src_corr, ref_corr ], -1)

    triangle_points = triangle_points.reshape(-1, 3)
    # print('triangle_points',triangle_points.shape)

    line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector(colors)
    # line_pcd.paint_uniform_color([1, 0.706, 0])
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)
    o3d.visualization.draw_geometries([line_pcd, src_pcd, ref_pcd], window_name='line_pcd src_pcd src_corr_pcd')

    # o3d.visualization.draw_geometries([src_corr_pcd, ref_pcd], window_name='src_corr_pcd  ref_pcd')
    # src_pcd.transform(transform)
    # src_corr_pcd.points = o3d.utility.Vector3dVector(weighted_ref)
    # o3d.visualization.draw_geometries([src_corr_pcd, src_pcd], window_name='src_corr_pcd src_pcd.transform(T)')
    #
    # ref_pcd.points = o3d.utility.Vector3dVector(ref)
    # o3d.visualization.draw_geometries([src_pcd, ref_pcd], window_name='src_pcd.transform(T)  ref_pcd')


def draw_registration_result(source, target, src_color, tgt_color):

    src_pcd = o3d.geometry.PointCloud()
    ref_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source)
    ref_pcd.points = o3d.utility.Vector3dVector(target)

    src_pcd.colors = o3d.utility.Vector3dVector(src_color)
    ref_pcd.colors = o3d.utility.Vector3dVector(tgt_color)

    # src_pcd.paint_uniform_color([1, 0.706, 0])
    # ref_pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([src_pcd, ref_pcd])

def draw_registration_result_no_blocking(source, target,vis):
    vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()

def get_npy_data(filename, index):
    all_data = np.load(filename, allow_pickle=True)
    # print(len(all_data))
    # xyz_src = torch.from_numpy(all_data[index * 3])
    # feat_src = torch.from_numpy(all_data[index * 3 + 2])
    # xyz_ref = torch.from_numpy(all_data[index * 3 + 3])
    # feat_ref = torch.from_numpy(all_data[index * 3 + 5])
    xyz = all_data[index * 4]
    normal = all_data[index * 4 + 1]
    feat = all_data[index * 4 + 2]
    color = all_data[index * 4 + 3]

    return xyz, normal, feat, color

def calGrad(point,normal,feature,kdTree):
    # n * 3; n * 3 ; n * d
    N = point.shape[0]
    d = feature.shape[1]
    grads = np.zeros([N,3,d])
    for i in range(N):
        pt = point[i,:].reshape(1,-1)
        nt = normal[i,:].reshape(1,-1)
        ft = feature[i,:].reshape(1,-1)
        _, idx = kdTree.query(pt, k=20, return_distance=True)
        # idx_ = np.reshape(idx,(-1,1))
        # neighbor_ = point[idx_, :]
        # neighbor = np.reshape(neighbor_, (N,-1, 3))
        neighbor_pt = point[idx, :].reshape(-1,3)
        neighbor_ft = feature[idx,:].reshape(-1,d)
        proj_pt = neighbor_pt - (neighbor_pt - pt) @ nt.T * nt
        A = proj_pt - pt
        b = neighbor_ft - ft
        A = np.concatenate((A,nt),axis=0)
        b = np.concatenate((b,np.zeros(d).reshape(1,d)))
        x = np.linalg.inv(A.T@A)@A.T@b
        grads[i,:,:] = x
    return  grads

def pt2plTrans(source,target,corr, weights):
    ps = source.point[corr[:, 0], :]
    pt = target.point[corr[:, 1], :]
    nt = target.normal[corr[:, 1], :]
    geo_A = np.concatenate((np.cross(ps, nt), nt), axis=1) * weights
    geo_b = np.sum((ps-pt)*nt, axis=1,keepdims=True) * weights

    Ja = geo_A
    res = geo_b

    vecTrans = -np.linalg.inv(Ja.T@Ja)@Ja.T@res
    vecTrans = np.squeeze(vecTrans)
    cx = np.cos(vecTrans[0])
    cy = np.cos(vecTrans[1])
    cz = np.cos(vecTrans[2])
    sx = np.sin(vecTrans[0])
    sy = np.sin(vecTrans[1])
    sz = np.sin(vecTrans[2])
    R = np.array([[cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz],
                  [cy*sz, cx*cz+sx*sy*sz, cx*sy*sz-sx*cz],
                  [-sy,            sx*cy,          cx*cy]])
    t = vecTrans[3:]
    transform = np.identity(4)
    transform[0:3, 0:3] = R
    transform[0:3, 3] = t
    t = t.reshape(3, 1)
    return  R, t, transform

class PointCloud:
    def __init__(self,point,normal,feature):
        self.point = point
        self.normal = normal
        self.feature = feature

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
    file_path = '/Bill/DataSet/RedWood/loft/'
    save_path = 'loft/ours/src2ref'
    end = 252

    # file_path = '/Bill/DataSet/RedWood/lobby/'
    # save_path = 'lobby/ours/src2ref'
    # end = 199
    # file_path = 'D:\PointCloud_DataSet\RedWood\\loft\\loft\\'
    # file_path = '/Bill/DataSet/RedWood/apartment/'
    # save_path = 'apartment/ours/src2ref'
    # end = 319

    # file_path = '/Bill/DataSet/RedWood/bedroom/'
    # save_path = 'bedroom/ours/src2ref'
    # end = 219

    # file_path = '/Bill/DataSet/RedWood/boardroom/'
    # save_path = 'boardroom/ours/src2ref'
    # end = 243

    groud_truth = file2matrix(file_path + 'reg_output.log')
    # print(len(groud_truth))
    err_R = []
    err_T = []
    trans_all = []
    fail_list = []
    start = 0
    # end = 244
    for j in range(start, end):
        print(
            'j',j
        )
        # index_src = j + 1
        # index_ref = j

        index_src = j
        index_ref = j + 1

        source_show = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_src))
        target_show = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_ref))
        filename = file_path + 'xyz_nor_feat_color.npy'

        xyz_src, normal_src, feat_src, color_src = get_npy_data(filename, index_src)
        xyz_ref, normal_ref, feat_ref, color_ref = get_npy_data(filename, index_ref)

        # draw_registration_result(xyz_src, xyz_ref, color_src, color_ref)
        # print('feat_src', feat_src.shape, feat_ref.shape)
        total_trans = np.eye(4)
        lambda_hybrid = 0.8
        lambda_color_ge = 0
        fail_flag = 0

        for i in range(30):
            src_hybrid_feature = np.concatenate(((lambda_hybrid) * feat_src,
                                                 ((1 - lambda_hybrid) * lambda_color_ge) * color_src,
                                                 ((1 - lambda_hybrid) * (1 - lambda_color_ge)) * xyz_src), 1)
            ref_hybrid_feature = np.concatenate(((lambda_hybrid) * feat_ref,
                                                 ((1 - lambda_hybrid) * lambda_color_ge) * color_ref,
                                                 ((1 - lambda_hybrid) * (1 - lambda_color_ge)) * xyz_ref), 1)
            # src_hybrid_feature = np.concatenate((np.sqrt(lambda_hybrid) * feat_src, np.sqrt((1-lambda_hybrid) * lambda_color_ge) * color_src, np.sqrt((1-lambda_hybrid) * (1-lambda_color_ge)) * xyz_src), 1)
            # ref_hybrid_feature = np.concatenate((np.sqrt(lambda_hybrid) * feat_ref, np.sqrt((1-lambda_hybrid) * lambda_color_ge) * color_ref, np.sqrt((1-lambda_hybrid) * (1-lambda_color_ge)) * xyz_ref), 1)
            feat_ref_tree = KDTree(ref_hybrid_feature)
            dist_feat, corr = feat_ref_tree.query(src_hybrid_feature, k = 1, return_distance = True)#src 找 tgt里边最近的点，得到的是tgt里面的索引
            # print('dist_feat',dist_feat.shape)
            corr_xyz_ref = xyz_ref[corr].reshape(-1,3)
            corr_xyz_src = xyz_src
            distance_threshold = np.sqrt(lambda_hybrid ** 2 * 0.4 + ((1-lambda_hybrid) * lambda_color_ge) ** 2 * 0.3 + ((1 - lambda_hybrid) * (1-lambda_color_ge)) ** 2 * 0.3 )
            ref_correct_corr = corr[dist_feat < distance_threshold]#满足距离要求的位置为1，然后再给对应关系，就得到ref中计算的点
            ref_correct_xyz = xyz_ref[ref_correct_corr]
            ref_correct_normal = normal_ref[ref_correct_corr]
            ref_correct_color = color_ref[ref_correct_corr]
            if ref_correct_xyz.shape[0] == 0:
                fail_flag = 1
                continue

            src_correct_corr = np.where((np.array(dist_feat < distance_threshold) > 0 ).reshape(-1, 1))[0]#因为src就是从0到n的索引，大于0是取了那些满足要求的位置，所以只需要知道dist_feat的哪个位置满足要求即可
            src_correct_xyz = xyz_src[src_correct_corr]
            src_correct_normal = normal_src[src_correct_corr]
            src_correct_color = color_src[src_correct_corr]

            source = PointCloud(src_correct_xyz, src_correct_normal, src_correct_color)
            target = PointCloud(ref_correct_xyz, ref_correct_normal, ref_correct_color)

            useful_dis = dist_feat[src_correct_corr]#这个距离向量是src和ref的距离，所以取src，假设你src第4个点满足要求，肯定是对应dist_feat中的第四个值嘛
            # show_point_cloud(corr_xyz_src, src_correct_xyz, xyz_ref, ref_correct_xyz)
            # weights = np.ones(src_correct_xyz.shape[0]).reshape(-1,1)#这里得到的就是满足要求的索引np.sum(np.power((src_correct_color - ref_correct_color), 2), 1).reshape(-1,1) *

            weights = np.exp(-useful_dis/0.1).reshape(-1,1)#这里得到的就是满足要求的索引
            weights = weights/np.sum(weights)
            # print('corr_xyz_ref',i , distance_threshold, ref_correct_corr.shape, xyz_src.shape, xyz_ref.shape, weights.shape,src_correct_corr.shape)

            N = src_correct_xyz.shape[0]
            corr_src = np.array(range(N)).reshape(N, 1)
            corr = np.concatenate((corr_src, corr_src), axis=1)#因为把有效的点都合在一起了
            R, t, transform = pt2plTrans(source, target, corr, weights)# 1 - 0.002 * i
            xyz_src = (R @ xyz_src.T + t).T
            source_show.transform(transform)
            lambda_hybrid = 0.9 * lambda_hybrid
            total_trans = transform @ total_trans
        if fail_flag == 1:
            total_trans = np.eye(4)
            fail_list.append(j)
            print('fail', j)

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
        print('err_R err_T', err_R[j - start], err_T[j - start],total_trans)
    if index_src > index_ref:
    #
    #     location = str(start) + '_' + str(end)

        err_all = [err_R, err_T]
        plt.figure("ERR_R ref2src")  # 图像窗口名称
        plt.plot(err_R)
        plt.savefig(save_path + '/%s_%s_err_All_ref2src.jpg'%(start, end))
        # plt.show()
        plt.figure("ERR_T ref2src")  # 图像窗口名称
        plt.plot(err_T)
        plt.savefig(save_path + '/%s_%s_trans_all_ref2src.jpg' % (start, end))
        # plt.show()
        np.savetxt(save_path + '/%s_%s_fail_list_ref2src.txt'%(start, end), fail_list)
        np.save(save_path + '/%s_%s_err_All_ref2src.npy'%(start, end), err_all)
        np.savetxt(save_path + '/%s_%s_err_All_ref2src.txt' % (start, end), err_all)
        np.save(save_path + '/%s_%s_trans_all_ref2src.npy'%(start, end), trans_all)
        np.savetxt(save_path + '/%s_%s_trans_all_ref2src.txt'%(start, end), np.array(trans_all).reshape(-1,4),fmt='%0.8f')

    else:
        err_all = [err_R, err_T]
        plt.figure("ERR_R src2ref")  # 图像窗口名称
        plt.plot(err_R)
        plt.savefig(save_path +  '/%s_%serr_All_src2ref.jpg'%(start, end))
        # plt.show()
        plt.figure("ERR_T src2ref")  # 图像窗口名称
        plt.plot(err_T)
        plt.savefig(save_path + '/%s_%strans_all_src2ref.jpg' % (start, end))
        # plt.show()
        np.savetxt(save_path + '/%s_%s_fail_list_src2ref.txt'%(start, end), fail_list)
        np.savetxt(save_path + '/%s_%serr_All_src2ref.txt' % (start, end), err_all)
        np.save(save_path + '/%s_%serr_All_src2ref.npy'%(start, end), err_all)
        np.save(save_path + '/%s_%strans_all_src2ref.npy'%(start, end), trans_all)
        np.savetxt(save_path + '/%s_%strans_all_src2ref.txt'%(start, end), np.array(trans_all).reshape(-1,4),fmt='%0.8f')


