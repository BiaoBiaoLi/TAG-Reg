import json
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

def draw_line_registration_result_no_blocking(source, target, lineset, lineset2, vis):
    vis.update_geometry(target)
    vis.update_geometry(source)
    vis.update_geometry(lineset)
    vis.update_geometry(lineset2)
    vis.poll_events()
    vis.update_renderer()

def draw_registration_result_no_blocking(source, vis):
    vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()

def get_npy_data(filename, index):
    all_data = np.load(filename, allow_pickle=True)
    print(len(all_data))
    # xyz_src = torch.from_numpy(all_data[index * 3])
    # feat_src = torch.from_numpy(all_data[index * 3 + 2])
    # xyz_ref = torch.from_numpy(all_data[index * 3 + 3])
    # feat_ref = torch.from_numpy(all_data[index * 3 + 5])
    xyz = all_data[index * 4]
    normal = all_data[index * 4 + 1]
    feat = all_data[index * 4 + 2]
    color = all_data[index * 4 + 3]

    return xyz, normal, feat, color

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

def line_point_cloud(src_corr_, ref_corr_, line_pcd, colors):
    # src = src_.copy()
    src_corr = copy.deepcopy(src_corr_)
    # ref = ref_.copy()
    ref_corr =  copy.deepcopy(ref_corr_)
    # ref[:,0] = ref[:,0] + 2.5
    ref_corr[:,0] = ref_corr[:,0] + 2.5
    temp_points = np.array(target_show.points)
    # temp_points[:,0] = temp_points[:,0] + 2.5
    # target_show.points = o3d.utility.Vector3dVector(temp_points)

    src_pcd = o3d.geometry.PointCloud()
    src_corr_pcd = o3d.geometry.PointCloud()

    ref_pcd = o3d.geometry.PointCloud()
    ref_corr_pcd = o3d.geometry.PointCloud()

    src_corr_pcd.points = o3d.utility.Vector3dVector(src_corr)
    src_corr_pcd.paint_uniform_color(colors)

    ref_corr_pcd.points = o3d.utility.Vector3dVector(ref_corr )
    ref_corr_pcd.paint_uniform_color(colors)

    # source_show.paint_uniform_color([1, 0.706, 0])
    # target_show.paint_uniform_color([0, 0.651, 0.929])

    ref_pcd.paint_uniform_color(colors)  # 蓝色
    src_corr_pcd.paint_uniform_color(colors)  # 黄色
    # src_pcd.paint_uniform_color([0, 0.651, 0.929])  # 红色

    line_size = src_corr.shape[0]
    line_src = np.arange(0, 2 * line_size, 2)  # 这个代表所有偶数
    rand_idxs = np.random.choice(line_size, math.ceil(line_size / 3), replace=False)
    #     print('line_src',line_src)
    line_src = line_src[rand_idxs].reshape(rand_idxs.shape[0], 1)
    #     print('line_src',line_src)
    line_ref = line_src + 1
    #     print('line_ref',line_ref)
    lines = np.concatenate([line_src, line_ref], -1).reshape(-1, 2)
    # lines = np.concatenate([line_ref, line_src], -1).reshape(-1, 2)
    #     print('lines',lines)
    # triangle_points=np.concatenate([data['points_ref'][1, :, :3].detach().cpu().numpy()+1,data['points_src'][1, :, :3].detach().cpu().numpy()],-1)

    triangle_points = np.concatenate([src_corr, ref_corr], -1)


    triangle_points = triangle_points.reshape(-1, 3)
    # print('triangle_points',triangle_points.shape)

    # line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    # line_pcd.colors = o3d.utility.Vector3dVector([colors])
    line_pcd.paint_uniform_color(colors)
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)
    # vis.add_geometry(line_pcd)
    return  line_pcd, src_corr_pcd, ref_corr_pcd
    # vis.add_geometry(src_pcd)
    # vis.add_geometry(ref_pcd)
    # o3d.visualization.draw_geometries([line_pcd, src_pcd, ref_pcd, source_show, target_show], window_name='line_pcd src_pcd src_corr_pcd')

def first_step_feature_extra_show_point_cloud(src_corr_, ref_corr_, line_pcd, colors, vis, source_show, target_show,
                                              src_corr_wr,
                                              ref_corr_wr, line_pcd_w, colors_w, weights, index):
    src_corr = copy.deepcopy(src_corr_)
    ref_corr = copy.deepcopy(ref_corr_)

    src_corr_w = copy.deepcopy(src_corr_wr)
    ref_corr_w = copy.deepcopy(ref_corr_wr)
    # source_show = copy.deepcopy(source_show_)
    ctr = vis.get_view_control()
    vis.add_geometry(source_show)

    with open("./look_at.json", 'r') as load_f:
        load_dict = json.load(load_f)
    # print(load_dict)
    # print(load_dict['trajectory'])
    trajectory = load_dict['trajectory'][0]
    print(trajectory['lookat'])
    ctr.set_lookat(trajectory['lookat'])

    ctr.change_field_of_view(trajectory['field_of_view'])
    ctr.set_up(trajectory['up'])
    ctr.set_zoom(trajectory['zoom'])
    ctr.set_front(trajectory['front'])
    if index < 6:
        src_corr[:, 0] = src_corr[:, 0] - 2.5 + index * 0.5
        src_corr_w[:, 0] = src_corr_w[:, 0] - 2.5 + index * 0.5

        temp_points = np.array(source_show.points)
        temp_points[:, 0] = temp_points[:, 0] - 2.5 + index * 0.5
        source_show.points = o3d.utility.Vector3dVector(temp_points)

    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(1)
    source_show_tree = KDTree(np.array(source_show.points))
    target_show_tree = KDTree(np.array(target_show.points))

    dist_feat, corr_corr1 = source_show_tree.query(src_corr, k=20,
                                                   return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引
    np.asarray(source_show.colors)[corr_corr1.reshape(-1), :] = [0, 0, 0]

    dist_feat, corr_corr2 = target_show_tree.query(ref_corr, k=20,
                                                   return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引
    np.asarray(target_show.colors)[corr_corr2.reshape(-1), :] = [0, 0, 0]

    vis.update_geometry(target_show)
    vis.poll_events()
    vis.update_renderer()
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    # vis.run()
    # vis.remove_geometry(source_show)


def first_show_line_point_cloud(src_, ref_, src_corr_, ref_corr_, line_pcd, colors, vis,source_show, target_show, src_corr_wr, ref_corr_wr, line_pcd_w, colors_w, weights, index):

    src_all = copy.deepcopy(src_)
    ref_all =  copy.deepcopy(ref_)

    src_corr = copy.deepcopy(src_corr_)
    ref_corr =  copy.deepcopy(ref_corr_)

    src_corr_w = copy.deepcopy(src_corr_wr)
    ref_corr_w =  copy.deepcopy(ref_corr_wr)
    # source_show =  copy.deepcopy(source_show_)
    ctr = vis.get_view_control()
    vis.add_geometry(source_show)
    vis.update_geometry(target_show)
    vis.poll_events()
    vis.update_renderer()
    # ctr.change_field_of_view(90)
    # ctr.set_lookat([ 0.96439201830872123, 2.1652333501072283, 2.0034809596565415 ])
    # ctr.set_up([ -0.017283470528902202, -0.94806581516680843, 0.31760430059835715 ])
    # ctr.set_zoom(1.1200000000000003)
    # ctr.set_front([ -0.040377242881946954, -0.31673081415929505, -0.94765567038837606 ])

    with open("./look_at.json", 'r') as load_f:
        load_dict = json.load(load_f)
    # print(load_dict)
    # print(load_dict['trajectory'])
    trajectory = load_dict['trajectory'][0]
    print(trajectory['lookat'])
    ctr.set_lookat(trajectory['lookat'])

    ctr.change_field_of_view(trajectory['field_of_view'])
    ctr.set_up(trajectory['up'])
    ctr.set_zoom(trajectory['zoom'])
    ctr.set_front(trajectory['front'])
    # ref[:,0] = ref[:,0] + 2.5
    if index<6:
        src_corr[:,0] = src_corr[:,0] - 2.5 + index * 0.5
        src_corr_w[:,0] = src_corr_w[:,0] - 2.5 + index * 0.5

        src_all[:,0] = src_all[:,0] - 2.5 + index * 0.5

        temp_points = np.array(source_show.points)
        temp_points[:,0] = temp_points[:,0] - 2.5 + index * 0.5
        source_show.points = o3d.utility.Vector3dVector(temp_points)
    else:
        src_corr[:, 0] = src_corr[:, 0]
        src_corr_w[:, 0] = src_corr_w[:, 0]

        temp_points = np.array(source_show.points)
        temp_points[:, 0] = temp_points[:, 0]
        source_show.points = o3d.utility.Vector3dVector(temp_points)
    print('ddddd')
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(1)
    source_show_tree = KDTree(np.array(source_show.points))
    target_show_tree = KDTree(np.array(target_show.points))

    dist_feat, corr_src_all = source_show_tree.query(src_all, k=20,
                                                   return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引
    np.asarray(source_show.colors)[corr_src_all.reshape(-1), :] = [0, 0, 0]

    dist_feat, corr_ref_all = target_show_tree.query(ref_all, k=20,
                                                   return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引
    np.asarray(target_show.colors)[corr_ref_all.reshape(-1), :] = [0, 0, 0]
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(1)
    source_show.paint_uniform_color([1, 0.706, 0])
    target_show.paint_uniform_color([0, 0.651, 0.929])
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)

    # for m in range(100000):
    #     print('googojgjgjioogogj')
    src_corr_pcd = o3d.geometry.PointCloud()
    ref_corr_pcd = o3d.geometry.PointCloud()
    src_corr_pcd.points = o3d.utility.Vector3dVector(src_corr)
    ref_corr_pcd.points = o3d.utility.Vector3dVector(ref_corr )
    ref_corr_pcd.paint_uniform_color(colors)
    src_corr_pcd.paint_uniform_color(colors)  # 黄色

    vis.get_render_option().point_size = 5

    ############################################################################################################
    line_size = src_corr.shape[0]
    line_src = np.arange(0, 2 * line_size, 2)  # 这个代表所有偶数
    # rand_idxs = np.random.choice(line_size, math.ceil(line_size / 3), replace=False)
    rand_idxs = np.random.choice(line_size,
                                 math.ceil(1000 * src_corr.shape[0] / (src_corr_w.shape[0] + src_corr.shape[0])),
                                 replace=False)
    line_src = line_src[rand_idxs].reshape(rand_idxs.shape[0], 1)
    line_ref = line_src + 1
    lines = np.concatenate([line_src, line_ref], -1).reshape(-1, 2)

    triangle_points = np.concatenate([src_corr, (src_corr + (ref_corr + src_corr) / 2) / 2], -1)
    triangle_points = triangle_points.reshape(-1, 3)
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.paint_uniform_color(colors)
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    line_size_w = src_corr_w.shape[0]
    line_src_w = np.arange(0, 2 * line_size_w, 2)  # 这个代表所有偶数
    # rand_idxs_w = np.random.choice(line_size_w, math.ceil(line_size_w / 3), replace=False)
    rand_idxs_w = np.random.choice(line_size_w,
                                   math.ceil(1000 * src_corr_w.shape[0] / (src_corr_w.shape[0] + src_corr.shape[0])),
                                   replace=False)
    line_src_w = line_src_w[rand_idxs_w].reshape(rand_idxs_w.shape[0], 1)
    line_ref_w = line_src_w + 1
    lines_w = np.concatenate([line_src_w, line_ref_w], -1).reshape(-1, 2)

    source_show_tree = KDTree(np.array(source_show.points))
    target_show_tree = KDTree(np.array(target_show.points))
    # print('ref_hybrid_feature',ref_hybrid_feature.shape,src_hybrid_feature.shape)
    index_src_correct = line_src / 2
    index_src_correct = index_src_correct.reshape(rand_idxs.shape[0]).astype(int)
    print(index_src_correct, src_corr[index_src_correct])
    dist_feat, corr_corr1 = source_show_tree.query(src_corr[index_src_correct], k=20,
                                                   return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引

    np.asarray(source_show.colors)[corr_corr1.reshape(-1), :] = [0, 0, 0]
    index_ref_correct = line_src / 2
    index_ref_correct = index_ref_correct.reshape(rand_idxs.shape[0]).astype(int)
    dist_feat, corr_corr2 = target_show_tree.query(ref_corr[index_ref_correct], k=20,
                                                   return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引

    index_src_wrong = line_src_w / 2
    index_src_wrong = index_src_wrong.reshape(rand_idxs_w.shape[0]).astype(int)
    dist_feat, corr_wrong1 = source_show_tree.query(src_corr_w[index_src_wrong], k=20,
                                                    return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引

    np.asarray(source_show.colors)[corr_wrong1.reshape(-1), :] = [0, 0, 0]
    index_ref_wrong = line_src_w / 2
    index_ref_wrong = index_ref_wrong.reshape(rand_idxs_w.shape[0]).astype(int)
    dist_feat, corr_wrong2 = target_show_tree.query(ref_corr_w[index_ref_wrong], k=20,
                                                    return_distance=True)  # src 找 tgt里边最近的点，得到的是tgt里面的索引
    # np.asarray(target_show.colors)[corr_wrong2.reshape(-1), :] = [0, 0, 0]
    # draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    ############################################################################################################

    triangle_points_w = np.concatenate([src_corr_w, (src_corr_w + (ref_corr_w + src_corr_w)/2)/2], -1)
    triangle_points_w = triangle_points_w.reshape(-1, 3)
    line_pcd_w.lines = o3d.utility.Vector2iVector(lines_w)
    line_pcd_w.paint_uniform_color(colors_w)
    line_pcd_w.points = o3d.utility.Vector3dVector(triangle_points_w)

    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(0.2)

    triangle_points = np.concatenate([src_corr, (ref_corr + src_corr)/2], -1)
    triangle_points = triangle_points.reshape(-1, 3)
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.paint_uniform_color(colors)
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    triangle_points_w = np.concatenate([src_corr_w, (src_corr_w + ref_corr_w)/2], -1)
    triangle_points_w = triangle_points_w.reshape(-1, 3)
    line_pcd_w.lines = o3d.utility.Vector2iVector(lines_w)
    line_pcd_w.paint_uniform_color(colors_w)
    line_pcd_w.points = o3d.utility.Vector3dVector(triangle_points_w)
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(0.2)

    triangle_points = np.concatenate([src_corr, (ref_corr + (ref_corr + src_corr)/2)/2], -1)
    triangle_points = triangle_points.reshape(-1, 3)
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.paint_uniform_color(colors)
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    triangle_points_w = np.concatenate([src_corr_w, (ref_corr_w + (src_corr_w + ref_corr_w)/2)/2], -1)
    triangle_points_w = triangle_points_w.reshape(-1, 3)
    line_pcd_w.lines = o3d.utility.Vector2iVector(lines_w)
    line_pcd_w.paint_uniform_color(colors_w)
    line_pcd_w.points = o3d.utility.Vector3dVector(triangle_points_w)
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(0.2)

    triangle_points = np.concatenate([src_corr, ref_corr], -1)
    triangle_points = triangle_points.reshape(-1, 3)
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    triangle_points_w = np.concatenate([src_corr_w,ref_corr_w], -1)
    triangle_points_w = triangle_points_w.reshape(-1, 3)
    line_pcd_w.lines = o3d.utility.Vector2iVector(lines_w)
    line_pcd_w.paint_uniform_color(colors_w)
    line_pcd_w.points = o3d.utility.Vector3dVector(triangle_points_w)

    np.asarray(target_show.colors)[corr_wrong2.reshape(-1), :] = [0, 0, 0]
    np.asarray(target_show.colors)[corr_corr2.reshape(-1), :] = [0, 0, 0]


    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(1)

    triangle_points = np.concatenate([src_corr, ref_corr], -1)
    triangle_points = triangle_points.reshape(-1, 3)
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.paint_uniform_color([0, 1, 0])
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    triangle_points_w = np.concatenate([src_corr_w,ref_corr_w], -1)
    triangle_points_w = triangle_points_w.reshape(-1, 3)
    line_pcd_w.lines = o3d.utility.Vector2iVector(lines_w)
    line_pcd_w.paint_uniform_color([1, 0, 0])
    line_pcd_w.points = o3d.utility.Vector3dVector(triangle_points_w)

    ############################################################################################################
    np.asarray(source_show.colors)[corr_corr1.reshape(-1), :] = [0, 1, 0]
    np.asarray(target_show.colors)[corr_corr2.reshape(-1), :] = [0, 1, 0]

    np.asarray(source_show.colors)[corr_wrong1.reshape(-1), :] = [1, 0, 0]
    np.asarray(target_show.colors)[corr_wrong2.reshape(-1), :] = [1, 0, 0]

    vis.update_geometry(target_show)
    vis.poll_events()
    vis.update_renderer()
    ############################################################################################################
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(1.5)

    triangle_points = np.concatenate([src_corr, ref_corr], -1)
    triangle_points = triangle_points.reshape(-1, 3)
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.paint_uniform_color([0, 1, 0])
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

    triangle_points_w = []
    line_pcd_w.lines = o3d.utility.Vector2iVector([])
    line_pcd_w.paint_uniform_color([1, 1, 1])
    line_pcd_w.points = o3d.utility.Vector3dVector(triangle_points_w)

    ############################################################################################################
    np.asarray(source_show.colors)[corr_corr1.reshape(-1), :] = [0, 1, 0]
    np.asarray(target_show.colors)[corr_corr2.reshape(-1), :] = [0, 1, 0]

    np.asarray(source_show.colors)[corr_wrong1.reshape(-1), :] = [1, 0.706, 0]
    np.asarray(target_show.colors)[corr_wrong2.reshape(-1), :] = [0, 0.651, 0.929]

    ############################################################################################################

    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    time.sleep(1.5)
    # vis.run()
    weight_colors = np.zeros((src_corr.shape[0],3))
    weight_colors[:,1] = 1
    weight_colors[:, 0] = 1 - weights[:, 0] * 500
    weight_colors[:, 2] = 1 - weights[:, 0] * 500
    # weight_colors = np.random.random((src_corr.shape[0],3))
    triangle_points = np.concatenate([src_corr, ref_corr], -1)
    triangle_points = triangle_points.reshape(-1, 3)
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector(weight_colors[[rand_idxs],:].reshape(rand_idxs.shape[0], 3))
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)
    draw_line_registration_result_no_blocking(source_show, target_show, line_pcd, line_pcd_w, vis)
    # vis.run()
    target_show.paint_uniform_color([0, 0.651, 0.929])
    vis.update_geometry(target_show)
    vis.poll_events()
    vis.update_renderer()
    # vis.remove_geometry(source_show)
    temp_points = np.array(source_show.points)
    temp_points[:, 0] = temp_points[:, 0] + 2.5 - index * 0.5
    source_show.points = o3d.utility.Vector3dVector(temp_points)
    vis.remove_geometry(line_pcd)
    vis.remove_geometry(line_pcd_w)
    source_show.paint_uniform_color([1, 0.706, 0])

    return  line_pcd, src_corr_pcd, ref_corr_pcd

def other_step_show_point_cloud(src_corr_, ref_corr_, line_pcd, colors, vis,source_show, target_show, src_corr_wr, ref_corr_wr, line_pcd_w, colors_w, weights, index, source_show_color, target_show_color):

    src_corr = copy.deepcopy(src_corr_)
    ref_corr =  copy.deepcopy(ref_corr_)

    src_corr_w = copy.deepcopy(src_corr_wr)
    ref_corr_w =  copy.deepcopy(ref_corr_wr)
    # source_show =  copy.deepcopy(source_show_)
    # vis.add_geometry(source_show)

    ctr = vis.get_view_control()
    # ctr.change_field_of_view(90)
    # ctr.set_lookat([ 0.96439201830872123, 2.1652333501072283, 2.0034809596565415 ])
    # ctr.set_up([ -0.017283470528902202, -0.94806581516680843, 0.31760430059835715 ])
    # ctr.set_zoom(1.1200000000000003)
    # ctr.set_front([ -0.040377242881946954, -0.31673081415929505, -0.94765567038837606 ])
    with open("./look_at.json", 'r') as load_f:
        load_dict = json.load(load_f)
    # print(load_dict)
    # print(load_dict['trajectory'])
    trajectory = load_dict['trajectory'][0]
    print(trajectory['lookat'])
    ctr.set_lookat(trajectory['lookat'])
    ctr.change_field_of_view(trajectory['field_of_view'])
    ctr.set_up(trajectory['up'])
    ctr.set_zoom(trajectory['zoom'])
    ctr.set_front(trajectory['front'])
    # ref[:,0] = ref[:,0] + 2.5
    if index<6:
        src_corr[:,0] = src_corr[:,0] - 2.5 + index * 0.5
        src_corr_w[:,0] = src_corr_w[:,0] - 2.5 + index * 0.5

        temp_points = np.array(source_show.points)
        temp_points[:,0] = temp_points[:,0] - 2.5 + index * 0.5
        source_show.points = o3d.utility.Vector3dVector(temp_points)

    draw_registration_result_no_blocking(source_show, vis)
    if index<6:
        temp_points = np.array(source_show.points)
        temp_points[:,0] = temp_points[:,0] + 2.5 - index * 0.5
        source_show.points = o3d.utility.Vector3dVector(temp_points)
    if i == 9 :
        source_show.colors = source_show_color.colors
        target_show.colors = target_show_color.colors
        vis.update_geometry(source_show)
        vis.update_geometry(target_show)
        vis.poll_events()
        vis.update_renderer()
        vis.run()



############################################################################################################

index_src = 38
index_ref = 39

index_ref = 53#53 54这一组得颜色信息加大为lambda_color_ge = 0.4
index_src = index_ref + 1

index_ref = 57#53 54这一组得颜色信息加大为lambda_color_ge = 0.4，前50帧基本可以lambda_color_ge = 0.1，lambda_hybrid = 0.8
index_src = index_ref + 1


# index_ref = 38
# index_src = index_ref + 1
dataset_names = ['loft', 'lobby', 'apartment', 'bedroom', 'boardroom','office1', 'office2', 'livingroom1', 'livingroom2']

index_src = 63
index_ref = 62
other_datasets = 0
file_path = 'D:\PointCloud_DataSet\RedWood\\loft\\'
file_path = 'D:\PointCloud_DataSet\RedWood\\lobby\\'
# file_path = 'D:\PointCloud_DataSet\RedWood\\apartment\\'
# file_path = 'D:\PointCloud_DataSet\RedWood\\boardroom\\'
# file_path = 'D:\PointCloud_DataSet\RedWood\livingroom1\\'
# file_path = 'D:\PointCloud_DataSet\RedWood\\bedroom\\'
# file_path = 'D:\PointCloud_DataSet\RedWood\\lobby\\'
# other_datasets_tru = 'D:\PointCloud_DataSet\RedWood\\office1\\office1'
# other_datasets_tru = 'D:\PointCloud_DataSet\RedWood\livingroom1-fragments-ply\livingroom1-evaluation\\gt.log'
#
# source_show = o3d.io.read_point_cloud(file_path + "cloud_bin_%s.ply"%(index_src))
# target_show = o3d.io.read_point_cloud(file_path + "cloud_bin_%s.ply"%(index_ref))

source_show = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_src))
target_show = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_ref))

source_show_color = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_src))
target_show_color = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_ref))

line_set_correct = o3d.geometry.LineSet()
line_set_wrong = o3d.geometry.LineSet()
line_set_all = o3d.geometry.LineSet()

filename = file_path + 'xyz_nor_feat_color.npy'

xyz_src, normal_src, feat_src, color_src = get_npy_data(filename, index_src)
xyz_ref, normal_ref, feat_ref, color_ref = get_npy_data(filename, index_ref)
source_show.estimate_normals()
target_show.estimate_normals()
# source_show.voxel_down_sample(0.05)
# target_show.voxel_down_sample(0.05)
# draw_registration_result(xyz_src, xyz_ref, color_src, color_ref)
# o3d.visualization.draw_geometries([target_show])
o3d.visualization.draw_geometries([target_show,source_show])
source_show.paint_uniform_color([1, 0.706, 0])
target_show.paint_uniform_color([0, 0.651, 0.929])
# print('feat_src', feat_src.shape, feat_ref.shape)
# o3d.visualization.draw_geometries([target_show,source_show])

total_trans = np.eye(4)
vis = o3d.visualization.Visualizer()#非阻塞显示
vis.create_window()
ctr = vis.get_view_control()
vis.get_render_option().point_size = 5
vis.add_geometry(source_show)
vis.add_geometry(target_show)
vis.add_geometry(line_set_correct)
vis.add_geometry(line_set_wrong)
vis.add_geometry(line_set_all)


lambda_color_ge = 0
start_time =  time.time()
groud_truth = file2matrix(file_path + 'reg_output.log')
# source_show.transform(groud_truth[index_src])

for i in range(0,10):
    lambda_hybrid = (np.sin(0.67*0.9**(i)*1.68))**2
    # lambda_hybrid = 1/(np.exp(i-5)+1)
    print('lambda_hybrid',lambda_hybrid)
    src_hybrid_feature = np.concatenate(((lambda_hybrid) * feat_src,
                                         ((1 - lambda_hybrid)) * xyz_src), 1)
    ref_hybrid_feature = np.concatenate(((lambda_hybrid) * feat_ref,
                                         ((1 - lambda_hybrid) * xyz_ref)), 1)
    # src_hybrid_feature = np.concatenate((np.sqrt(lambda_hybrid) * feat_src, np.sqrt((1-lambda_hybrid) * lambda_color_ge) * color_src, np.sqrt((1-lambda_hybrid) * (1-lambda_color_ge)) * xyz_src), 1)
    # ref_hybrid_feature = np.concatenate((np.sqrt(lambda_hybrid) * feat_ref, np.sqrt((1-lambda_hybrid) * lambda_color_ge) * color_ref, np.sqrt((1-lambda_hybrid) * (1-lambda_color_ge)) * xyz_ref), 1)
    feat_ref_tree = KDTree(ref_hybrid_feature)
    # print('ref_hybrid_feature',ref_hybrid_feature.shape,src_hybrid_feature.shape)
    dist_feat, corr = feat_ref_tree.query(src_hybrid_feature, k = 1, return_distance = True)#src 找 tgt里边最近的点，得到的是tgt里面的索引

    # print('dist_feat',dist_feat.shape)
    corr_xyz_ref = xyz_ref[corr].reshape(-1,3)
    corr_xyz_src = xyz_src
    # distance_threshold = 10
    distance_threshold = np.sqrt(lambda_hybrid ** 2 * 0.4 + ((1-lambda_hybrid) * lambda_color_ge) ** 2 * 0.5 + ((1 - lambda_hybrid) * (1-lambda_color_ge)) ** 2 * 0.3 )
    ref_correct_corr = corr[dist_feat < distance_threshold]#满足距离要求的位置为1，然后再给对应关系，就得到ref中计算的点
    ref_correct_xyz = xyz_ref[ref_correct_corr]
    ref_correct_normal = normal_ref[ref_correct_corr]
    ref_correct_color = color_ref[ref_correct_corr]

    ref_wrong_corr = corr[dist_feat >= distance_threshold]#满足距离要求的位置为1，然后再给对应关系，就得到ref中计算的点
    ref_wrong_xyz = xyz_ref[ref_wrong_corr]
    src_wrong_corr = np.where((np.array(dist_feat >= distance_threshold) > 0 ).reshape(-1, 1))[0]#因为src就是从0到n的索引，大于0是取了那些满足要求的位置，所以只需要知道dist_feat的哪个位置满足要求即可
    src_wrong_xyz = xyz_src[src_wrong_corr]

    src_correct_corr = np.where((np.array(dist_feat < distance_threshold) > 0 ).reshape(-1, 1))[0]#因为src就是从0到n的索引，大于0是取了那些满足要求的位置，所以只需要知道dist_feat的哪个位置满足要求即可
    src_correct_xyz = xyz_src[src_correct_corr]
    src_correct_normal = normal_src[src_correct_corr]
    src_correct_color = color_src[src_correct_corr]

    # vis.add_geometry(line_set_all)
    # vis.add_geometry(line_set_correct_src)
    # vis.add_geometry(line_set_correct_ref)
    # vis.add_geometry(line_set_wrong_src)
    # vis.add_geometry(line_set_wrong_ref)
    # vis.add_geometry(source_show)

    source = PointCloud(src_correct_xyz, src_correct_normal, src_correct_color)
    target = PointCloud(ref_correct_xyz, ref_correct_normal, ref_correct_color)

    useful_dis = dist_feat[src_correct_corr]#这个距离向量是src和ref的距离，所以取src，假设你src第4个点满足要求，肯定是对应dist_feat中的第四个值嘛
    # show_point_cloud(corr_xyz_src, src_correct_xyz, xyz_ref, ref_correct_xyz)
    # weights = np.ones(src_correct_xyz.shape[0]).reshape(-1,1)#这里得到的就是满足要求的索引np.sum(np.power((src_correct_color - ref_correct_color), 2), 1).reshape(-1,1) *

    weights = np.exp(-useful_dis/0.1).reshape(-1,1)#这里得到的就是满足要求的索引
    weights = weights/(np.sum(weights) + 0.000001)
    print('corr_xyz_ref',i , distance_threshold, ref_correct_corr.shape, xyz_src.shape, xyz_ref.shape, weights.shape,src_correct_corr.shape)
    if i==0:
        # first_step_feature_extra_show_point_cloud(xyz_src, xyz_ref, line_set_correct, [0, 0, 0], vis, source_show, target_show, src_wrong_xyz, ref_wrong_xyz,
        #                                                                           line_set_wrong, [0, 0, 0], weights, i)
        first_show_line_point_cloud(xyz_src, xyz_ref, src_correct_xyz, ref_correct_xyz, line_set_correct, [0, 0, 0], vis, source_show, target_show, src_wrong_xyz, ref_wrong_xyz,
                                                                                  line_set_wrong, [0, 0, 0], weights, i)

    else:
        other_step_show_point_cloud(src_correct_xyz, ref_correct_xyz, line_set_correct, [0, 0, 0], vis, source_show,
                                    target_show, src_wrong_xyz, ref_wrong_xyz,
                                    line_set_wrong, [0, 0, 0], weights, i, source_show_color, target_show_color)


    N = src_correct_xyz.shape[0]
    corr_src = np.array(range(N)).reshape(N, 1)
    corr = np.concatenate((corr_src, corr_src), axis=1)#因为把有效的点都合在一起了
    R, t, transform = pt2plTrans(source, target, corr, weights)# 1 - 0.002 * i
    xyz_src = (R @  xyz_src.T + t).T

    # lambda_hybrid = 0.9 * lambda_hybrid
    total_trans = transform @ total_trans
    source_show.transform(transform)
    source_show_color.transform(transform)

    # vis.add_geometry(line_set_wrong)
    # vis.add_geometry(line_set_correct)
    # if i < 9:
        # vis.remove_geometry(line_set_all)
        # vis.remove_geometry(line_set_correct_src)
        # vis.remove_geometry(line_set_correct_ref)
        # vis.remove_geometry(line_set_wrong_src)
        # vis.remove_geometry(line_set_wrong_ref)
        # vis.remove_geometry(source_show)
    if i < 5:
        time.sleep(0.2)
    else:
        time.sleep(0.1)

print('time',time.time() - start_time)

# print(total_trans)
# vis.run()
# vis.close()
total_trans = total_trans
R = total_trans[:3, :3].reshape(3, 3)
# t = total_trans[:3, 3].reshape(-1, 1)
tmpt = np.identity(4)
err_R = np.arccos((np.trace(R @ groud_truth[index_src][:3, :3]) - 1) / 2) * 180 / np.pi
err_T = (np.linalg.norm(-R.T @ t - groud_truth[index_src][:3, 3].reshape(-1, 1), ord=2, axis=0))
print('err_R', err_R, err_T,tmpt, total_trans)
source_show = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_src))
target_show = o3d.io.read_point_cloud(file_path + "mesh_%s.ply"%(index_ref))
temp_pcd = copy.deepcopy(source_show)
# o3d.visualization.draw_geometries([target_show,temp_pcd.transform(np.linalg.inv(tmpt))])
source_show.transform(total_trans)
# source_show.estimate_normals()
# target_show.estimate_normals()
# source_show.paint_uniform_color([1, 0.706, 0])
# target_show.paint_uniform_color([0, 0.651, 0.929])
# o3d.visualization.draw_geometries([target_show,source_show])
# draw_registration_result(xyz_src, xyz_ref, color_src, color_ref)
# source_show.paint_uniform_color([1, 0.706, 0])
# target_show.paint_uniform_color([0, 0.651, 0.929])
# o3d.visualization.draw_geometries([source_show, target_show], mesh_show_back_face=True)


# beta = 36.9520
# alpha = 0.4
# [[ 0.99936387 -0.03563891 -0.00131546 -0.04066989]
#  [ 0.0350326   0.97412     0.2233001   0.27904324]
#  [-0.00667676 -0.22320414  0.97474886  0.40576242]
#  [ 0.          0.          0.          1.        ]]

# [[ 0.99960782 -0.02786165  0.00281661 -0.04683648]
#  [ 0.02643835  0.97210381  0.23305619  0.2926286 ]
#  [-0.00923137 -0.23289033  0.97245919  0.42885402]
#  [ 0.          0.          0.          1.        ]]