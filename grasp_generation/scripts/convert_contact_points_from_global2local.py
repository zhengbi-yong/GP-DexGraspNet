import json
import os
import sys
import torch
import transforms3d
sys.path.append(os.path.realpath("../"))
from utils.config import get_abspath

# 获取一些需要使用的路径
scripts_directory = get_abspath(1, __file__)
graspgeneration_directory = get_abspath(2, __file__)
sys.path.append(graspgeneration_directory)

dexgraspnet_directory = get_abspath(3, __file__)
leaphandcentered_directory = os.path.join(graspgeneration_directory, "leaphand_centered")

def create_transform_matrix(xyz, rpy):
    """ 创建4x4的变换矩阵 """
    R = torch.tensor(transforms3d.euler.euler2mat(*rpy, axes='sxyz'), dtype=torch.float)
    T = torch.tensor(xyz, dtype=torch.float).view(3, 1)
    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T.squeeze()
    return transform_matrix

def global_to_local(global_points, transform_matrix):
    """ 将全局坐标点转换为局部坐标点 """
    local_points = []
    for point in global_points:
        global_point_hom = torch.ones(4)
        global_point_hom[:3] = torch.tensor(point)
        local_point_hom = torch.inverse(transform_matrix) @ global_point_hom
        local_points.append(local_point_hom[:3].numpy().tolist())
    return local_points

def get_full_transform(transform_matrices, parent_map, link):
    """ 递归地构建从root到当前link的完整变换矩阵 """
    if link not in parent_map or parent_map[link] == "":
        return transform_matrices[link]
    else:
        # 先应用父link的变换，然后应用当前link的变换
        return get_full_transform(transform_matrices, parent_map, parent_map[link]) @ transform_matrices[link]

def print_parent_child_relationship(parent_map):
    """ 打印父子关系 """
    for child, parent in parent_map.items():
        print(f"{parent} -> {child}")

def print_full_transform_order(transform_matrices, parent_map, link, order=[]):
    """ 递归地打印完整的变换顺序 """
    if link not in parent_map or parent_map[link] == "":
        print(" -> ".join(reversed(order + [link])))
    else:
        print_full_transform_order(transform_matrices, parent_map, parent_map[link], order + [link])

# 读取transforms_info.json文件
transforms_info_file = os.path.join(leaphandcentered_directory, 'transforms_info.json')
with open(transforms_info_file, 'r') as file:
    transforms_info = json.load(file)
# 从文件中读取transforms_info和link的父子关系
with open(os.path.join(leaphandcentered_directory, 'transforms_info.json'), 'r') as file:
    transforms_info = json.load(file)

with open(os.path.join(leaphandcentered_directory, 'parent_map.json'), 'r') as file:
    parent_map = json.load(file)
# 创建每个link的变换矩阵
transform_matrices = {link: create_transform_matrix(**info) for link, info in transforms_info.items()}

# 打印父子关系
print("Parent-Child Relationship:")
print_parent_child_relationship(parent_map)

# 为每个link计算相对于全局坐标系的完整变换矩阵
full_transform_matrices = {link: get_full_transform(transform_matrices, parent_map, link) for link in transform_matrices}

# 打印完整的变换顺序
print("\nFull Transform Order:")
for link in transform_matrices.keys():
    print_full_transform_order(transform_matrices, parent_map, link)

# 读取接触点数据
with open(os.path.join(leaphandcentered_directory, 'contact_points.json'), 'r') as file:
    contact_points = json.load(file)

# 应用完整的变换矩阵进行坐标转换
local_contact_points = {link: global_to_local(points, full_transform_matrices[link]) for link, points in contact_points.items()}

# 保存转换后的点
with open(os.path.join(leaphandcentered_directory, 'local_contact_points.json'), 'w') as file:
    json.dump(local_contact_points, file, indent=4)