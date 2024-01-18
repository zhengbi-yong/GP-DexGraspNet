# import json
# import os
# import sys
# sys.path.append(os.path.realpath("."))
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from utils.config import load_config, get_abspath
# scripts_directory = get_abspath(1,__file__)
# graspgeneration_directory = get_abspath(2,__file__)
# dexgraspnet_directory = get_abspath(3,__file__)
# leaphandcentered_directory = os.path.join(graspgeneration_directory, "leaphand_centered")
# debug_directory = os.path.join(graspgeneration_directory, "debug")
# optimize_process_directory = os.path.join(graspgeneration_directory, "debug/optimize_process")
# def load_json_data(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# def create_plotly_scatter3d(data, marker_color='blue'):
#     flattened_data = [point for sublist in data for point in sublist]
#     x, y, z = zip(*flattened_data)
#     scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=marker_color, size=5))
#     return scatter

# def save_plot_as_html(data, file_name, output_folder):
#     scatter_plot = create_plotly_scatter3d(data)
#     fig = go.Figure()
#     fig.add_trace(scatter_plot)
#     fig.update_layout(scene_aspectmode='data')
#     output_path = os.path.join(output_folder, file_name + '.html')
#     fig.write_html(output_path)
# # 文件夹路径
# folder_path = optimize_process_directory  # 请将此路径更改为实际文件夹的路径
# output_folder = f'{debug_directory}/optimize_process_visualization' # 修改为输出文件夹路径
# # 获取所有JSON文件
# files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
# # 为每个文件创建一个子图
# for file in files:
#     file_path = os.path.join(folder_path, file)
#     json_data = load_json_data(file_path)
#     save_plot_as_html(json_data, file.split('.')[0], output_folder)
import json
import os
import sys
import plotly.graph_objects as go
import numpy as np

sys.path.append(os.path.realpath("."))
from utils.config import load_config, get_abspath
scripts_directory = get_abspath(1,__file__)
graspgeneration_directory = get_abspath(2,__file__)
dexgraspnet_directory = get_abspath(3,__file__)
leaphandcentered_directory = os.path.join(graspgeneration_directory, "leaphand_centered")
debug_directory = os.path.join(graspgeneration_directory, "debug")
optimize_process_directory = os.path.join(graspgeneration_directory, "debug/optimize_process")
optimize_process_json_directory = os.path.join(optimize_process_directory, "jsondata")
optimize_process_obj_directory = os.path.join(optimize_process_directory, "objdata")
optimize_process_obj_hand_directory = os.path.join(optimize_process_obj_directory, "hand")
optimize_process_obj_object_directory = os.path.join(optimize_process_obj_directory, "object")

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_plotly_scatter3d(data, marker_color='blue'):
    flattened_data = [point for sublist in data for point in sublist]
    x, y, z = zip(*flattened_data)
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=marker_color, size=5))
    return scatter

def load_obj_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        vertices = []
        faces = []
        for line in lines:
            if line.startswith('v '):
                vertices.append(np.array(line.split()[1:], dtype=float))
            elif line.startswith('f'):
                face = [int(face.split('/')[0]) for face in line.split()[1:]]
                faces.append(np.array(face, dtype=int) - 1)  # OBJ文件的索引从1开始
        return np.array(vertices), np.array(faces)

def create_plotly_mesh3d(vertices, faces, color='lightblue', opacity=0.5):
    x, y, z = vertices.T
    i, j, k = faces.T
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity)
    return mesh

def save_plot_as_html(contact_data, mesh_data, file_name, output_folder):
    scatter_plot = create_plotly_scatter3d(contact_data)
    mesh_plot = create_plotly_mesh3d(*mesh_data)
    fig = go.Figure()
    fig.add_trace(scatter_plot)
    fig.add_trace(mesh_plot)
    fig.update_layout(scene_aspectmode='data')
    output_path = os.path.join(output_folder, file_name + '.html')
    fig.write_html(output_path)

# 文件夹路径
contact_points_folder = optimize_process_json_directory
hand_mesh_folder = optimize_process_obj_hand_directory
output_folder = os.path.join(debug_directory, 'optimize_process_visualization')
# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有JSON和OBJ文件
contact_files = [f for f in os.listdir(contact_points_folder) if f.endswith('.json')]
mesh_files = [f for f in os.listdir(hand_mesh_folder) if f.endswith('.obj')]
# 为每个文件创建一个子图
for contact_file in contact_files:
    contact_file_path = os.path.join(contact_points_folder, contact_file)
    contact_data = load_json_data(contact_file_path)

    step = contact_file.split('_')[-1].split('.')[0]
    mesh_file = f'hand_mesh_step_{step}.obj'
    if mesh_file in mesh_files:
        mesh_file_path = os.path.join(hand_mesh_folder, mesh_file)
        mesh_data = load_obj_data(mesh_file_path)
        save_plot_as_html(contact_data, mesh_data, f'1visualization_step_{step}', output_folder)