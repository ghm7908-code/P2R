#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 3:06 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : building3d.py.py
# @IDE     : PyCharm

import os
import sys
import glob
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from collections import defaultdict


def load_wireframe(wireframe_file):
    import numpy as np
    
    vertices = []
    all_edges = []
    
    with open(wireframe_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            # 解析顶点: v x y z
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # 解析边: l v1 v2 v3 ...
            elif parts[0] == 'l':
                # OBJ 索引是从 1 开始的，需要减 1 转为 Python 的 0 索引
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                # 将连续点转为边对: [1, 2, 3] -> [[0, 1], [1, 2]]
                for i in range(len(indices) - 1):
                    all_edges.append([indices[i], indices[i+1]])
    
    # 转换为 numpy 数组
    vertices = np.array(vertices, dtype=np.float32)
    
    if len(all_edges) > 0:
        edges = np.array(all_edges, dtype=np.int32)
    else:
        # 如果文件里没有 l 标签，返回空数组避免崩溃
        edges = np.zeros((0, 2), dtype=np.int32)
            
    return vertices, edges


def save_wireframe(vertices, edges, wireframe_file):
    r"""
    :param wireframe_file: wireframe file name
    :param vertices: N * 3, vertex coordinates
    :param edges: M * 2,
    :return:
    """
    with open(wireframe_file, 'w') as f:
        for vertex in vertices:
            line = ' '.join(map(str, vertex))
            f.write('v ' + line + '\n')
        for edge in edges:
            edge = ' '.join(map(str, edge + 1))
            f.write('l ' + edge + '\n')



def random_sampling(pc, num_points, replace=None, return_choices=False):
    r"""
    :param pc: N * 3
    :param num_points: Int
    :param replace:
    :param return_choices:
    :return:
    """
    if replace is None:
        replace = pc.shape[0] < num_points
    choices = np.random.choice(pc.shape[0], num_points, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def farthest_point_sampling(pc, num_points):
    """
    最远点采样 (FPS)
    :param pc: N * 3 (or N * C) NumPy array
    :param num_points: 采样目标点数
    :return: 采样后的点云
    """
    N, C = pc.shape
    xyz = pc[:, 0:3] # 仅根据空间坐标计算距离
    centroids = np.zeros(num_points, dtype=np.int32)
    distance = np.ones(N) * 1e10
    
    # 随机选择第一个起始点
    farthest = np.random.randint(0, N)
    
    for i in range(num_points):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, 3)
        # 计算所有点到当前采样点的欧式距离
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        # 更新距离向量，保留每个点到已采样点集的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 下一个点选择距离当前点集最远的点
        farthest = np.argmax(distance)
        
    return pc[centroids]


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Building3DReconstructionDataset(Dataset):
    def __init__(self, dataset_config, split_set, logger=None):
        self.dataset_config = dataset_config
        self.roof_dir = dataset_config.root_dir
        self.num_points = dataset_config.num_points
        self.use_color = dataset_config.use_color
        self.use_intensity = dataset_config.use_intensity
        self.normalize = dataset_config.normalize
        self.augment = dataset_config.augment

        assert split_set in ["train", "val"]
        self.split_set = split_set

        self.pc_files, self.wireframe_files = self.load_files()

        if logger:
            logger.info("Total Sample: %d" % len(self.pc_files))

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, index):
        # ------------------------------- Point Clouds ------------------------------
        # load point clouds
        pc_file = self.pc_files[index]
        pc = np.loadtxt(pc_file, dtype=np.float64)

        # point clouds processing
        if not self.use_color:
            point_cloud = pc[:, 0:3]
        elif self.use_color and not self.use_intensity:
            point_cloud = pc[:, 0:7]
            point_cloud[:, 3:] = point_cloud[:, 3:] / 256.0
        elif not self.use_color and self.use_intensity:
            point_cloud = np.concatenate((pc[:, 0:3], pc[:, 7]), axis=1)
        else:
            point_cloud = pc
            point_cloud[:, 3:7] = point_cloud[:, 3:7] / 256.0

        # ------------------------------- Wireframe ------------------------------
        # load wireframe
        wireframe_file = self.wireframe_files[index]
        wf_vertices, wf_edges = load_wireframe(wireframe_file)

        # ------------------------------- Dataset Preprocessing ------------------------------
        if self.normalize:
            centroid = np.mean(point_cloud[:, 0:3], axis=0)
            point_cloud[:, 0:3] -= centroid
            max_distance = np.max(np.linalg.norm(point_cloud[:, 0:3], axis=1))
            point_cloud[:, 0:3] /= max_distance

            wf_vertices -= centroid
            wf_vertices /= max_distance

        if self.num_points:
            if point_cloud.shape[0] < self.num_points:
                point_cloud = random_sampling(point_cloud, self.num_points)
            else:
                point_cloud = farthest_point_sampling(point_cloud, self.num_points)

        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                wf_vertices[:, 0] = -1 * wf_vertices[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                wf_vertices[:, 1] = -1 * wf_vertices[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            wf_vertices[:, 0:3] = np.dot(wf_vertices[:, 0:3], np.transpose(rot_mat))

        # -------------------------------Edge Vertices ------------------------
        wf_edges_vertices = np.stack((wf_vertices[wf_edges[:, 0]], wf_vertices[wf_edges[:, 1]]), axis=1)
        wf_edges_vertices = wf_edges_vertices[
            np.arange(wf_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(wf_edges_vertices[:, :, -1]),
                                                                        axis=1)]
        wf_centers = (wf_edges_vertices[..., 0, :] + wf_edges_vertices[..., 1, :]) / 2
        wf_edge_number = wf_edges.shape[0]

        # ------------------------------- Return Dict ------------------------------
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['wf_vertices'] = wf_vertices.astype(np.float32)
        ret_dict['wf_edges'] = wf_edges.astype(np.int64)
        ret_dict['wf_centers'] = wf_centers.astype(np.float32)
        ret_dict['wf_edge_number'] = wf_edge_number
        ret_dict['wf_edges_vertices'] = wf_edges_vertices.reshape((-1, 6)).astype(np.float32)
        if self.normalize:
            ret_dict['centroid'] = centroid
            ret_dict['max_distance'] = max_distance
        ret_dict['scan_idx'] = np.array(os.path.splitext(os.path.basename(pc_file))[0]).astype(np.int64)
        return ret_dict

    @staticmethod
    def collate_batch(batch):
        from collections import defaultdict
        input_dict = defaultdict(list)
        for item in batch:
            for key, val in item.items():
                input_dict[key].append(val)

        ret_dict = {}
        batch_size = len(batch)
        ret_dict['batch_size'] = batch_size

        # --- 核心修复：添加归一化参数的聚合 ---
        if 'centroid' in input_dict:
            # 将列表中的 numpy 数组堆叠成 [B, 3] 的 Tensor
            ret_dict['centroid'] = torch.from_numpy(np.stack(input_dict['centroid'], axis=0))
        
        if 'max_distance' in input_dict:
            # 将列表中的标量转为 [B] 的 Tensor
            ret_dict['max_distance'] = torch.from_numpy(np.array(input_dict['max_distance']))
            
        # 方便 test.py 获取文件名
        if 'sample_id' in input_dict:
            ret_dict['sample_id'] = input_dict['sample_id']
        else:
            # 如果 __getitem__ 没写 sample_id，这里根据 scan_idx 补一个
            ret_dict['sample_id'] = [str(idx.item()) for idx in input_dict['scan_idx']]
        # ------------------------------------

        # 1. 处理点云 (保持原样)
        points_list = [torch.from_numpy(p) if isinstance(p, np.ndarray) else p for p in input_dict['point_clouds']]
        ret_dict['points'] = torch.cat(points_list, dim=0)
        ret_dict['xyz_batch_cnt'] = torch.IntTensor([len(p) for p in points_list])

        # 2. 处理关键点真值 (保持原样)
        vectors_list = [torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v.float() 
                        for v in input_dict['wf_vertices']]
        max_m = max([v.shape[0] for v in vectors_list])
        padded_vectors = torch.zeros((batch_size, max_m, 3))
        for i, v in enumerate(vectors_list):
            m = v.shape[0]
            padded_vectors[i, :m, :] = v
        ret_dict['vectors'] = padded_vectors 

        # 3. 处理边 (保持原样)
        edges_list = []
        vertex_offset = 0
        all_edges_stacked = []
        for i in range(batch_size):
            e = torch.from_numpy(input_dict['wf_edges'][i]).long()
            edges_list.append(e) 
            all_edges_stacked.append(e + vertex_offset)
            vertex_offset += vectors_list[i].shape[0]

        ret_dict['edges'] = edges_list 
        ret_dict['wf_edges'] = torch.cat(all_edges_stacked, dim=0)
        
        return ret_dict

    def load_files(self):
        import os
        import glob
        
        data_dir = os.path.join(self.roof_dir, self.split_set)
        # 获取所有 xyz 文件
        pc_files = sorted(glob.glob(os.path.join(data_dir, 'xyz', '*.xyz')))
        
        valid_pc = []   
        valid_wf = []
        
        for pc_p in pc_files:
            base = os.path.splitext(os.path.basename(pc_p))[0]
            # 兼容性：同时检查 .obj 和 .ply
            wf_obj = os.path.join(data_dir, 'wireframe', base + ".obj")
            wf_ply = os.path.join(data_dir, 'wireframe', base + ".ply")
            
            if os.path.exists(wf_obj):
                valid_pc.append(pc_p)
                valid_wf.append(wf_obj)
            elif os.path.exists(wf_ply):
                valid_pc.append(pc_p)
                valid_wf.append(wf_ply)
        
        # 调试信息：如果还是不行，这行能告诉你到底哪里空了
        if len(valid_pc) == 0:
            print(f"DEBUG: 在目录 {data_dir} 下未找到匹配的 xyz/wf 文件对")
            print(f"DEBUG: 扫描到的 xyz 数量: {len(pc_files)}")
            
        return valid_pc, valid_wf

    def print_self_values(self):
        attributes = vars(self)
        for attribute, value in attributes.items():
            print(attribute, "=", value)
