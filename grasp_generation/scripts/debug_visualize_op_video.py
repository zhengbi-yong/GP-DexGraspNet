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
                vertex_coords = line.split()[1:]
                vertices.append(np.array(vertex_coords, dtype=float))
            elif line.startswith('f'):
                face_indices = line.split()[1:]
                face = [int(index.split('/')[0]) for index in face_indices]
                faces.append(np.array(face, dtype=int) - 1)

        return np.array(vertices), np.array(faces)

def filter_invalid_faces(vertices, faces):
    """筛选掉包含无效顶点索引的面。"""
    valid_faces = []
    for face in faces:
        if np.all(face < len(vertices)):
            valid_faces.append(face)
        else:
            print(f"Skipping invalid face: {face}")
    return np.array(valid_faces)

def create_plotly_mesh3d(vertices, faces, color='lightblue', opacity=0.5):
    print(f"Creating mesh: {len(vertices)} vertices, {len(faces)} faces")
    print("Sample vertices:", vertices[:5])  # 输出前5个顶点作为样本
    print("Sample faces:", faces[:5])       # 输出前5个面作为样本

    # 筛选掉无效的面
    faces = filter_invalid_faces(vertices, faces)

    x, y, z = vertices.T
    i, j, k = faces.T

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=0.5)
    return mesh

# 在 save_plot_as_html 函数中也添加相应的 try...except 结构以捕获并输出异常
# def save_plot_as_html(contact_data, mesh_data, file_name, output_folder):
#     try:
#         scatter_plot = create_plotly_scatter3d(contact_data)
#         mesh_plot = create_plotly_mesh3d(*mesh_data)
#         fig = go.Figure()
#         fig.add_trace(scatter_plot)
#         fig.add_trace(mesh_plot)
#         fig.update_layout(scene_aspectmode='data')
#         output_path = os.path.join(output_folder, file_name + '.html')
#         fig.write_html(output_path)
#     except Exception as e:
#         print(f"Error while saving plot for {file_name}: {e}")
def save_plot_as_html(contact_data, hand_mesh_data, object_mesh_data, file_name, output_folder):
    try:
        scatter_plot = create_plotly_scatter3d(contact_data)
        hand_mesh_plot = create_plotly_mesh3d(*hand_mesh_data)
        object_mesh_plot = create_plotly_mesh3d(*object_mesh_data)
        fig = go.Figure()
        fig.add_trace(scatter_plot)
        fig.add_trace(hand_mesh_plot)
        fig.add_trace(object_mesh_plot)
        fig.update_layout(scene_aspectmode='data')
        output_path = os.path.join(output_folder, file_name + '.html')
        fig.write_html(output_path)
    except Exception as e:
        print(f"Error while saving plot for {file_name}: {e}")


# # 文件夹路径
# contact_points_folder = optimize_process_json_directory
# hand_mesh_folder = optimize_process_obj_hand_directory
# output_folder = os.path.join(debug_directory, 'optimize_process_visualization')
# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 获取所有JSON和OBJ文件
# contact_files = [f for f in os.listdir(contact_points_folder) if f.endswith('.json')]
# mesh_files = [f for f in os.listdir(hand_mesh_folder) if f.endswith('.obj')]

# # 为每个文件创建一个子图
# for contact_file in contact_files:
#     contact_file_path = os.path.join(contact_points_folder, contact_file)
#     contact_data = load_json_data(contact_file_path)
#     object_mesh_file = f'object_mesh_step_{step}_0.obj'  # 假设每一步都有一个名为 object_mesh_step_{step}_0.obj 的物体网格文件
#     object_mesh_file_path = os.path.join(optimize_process_obj_object_directory, object_mesh_file)

#     step = contact_file.split('_')[-1].split('.')[0]
#     mesh_file = f'hand_mesh_step_{step}.obj'
#     if mesh_file in mesh_files:
#         mesh_file_path = os.path.join(hand_mesh_folder, mesh_file)
#         mesh_data = load_obj_data(mesh_file_path)

#         print(f"Step {step}: Loaded {len(mesh_data[0])} vertices and {len(mesh_data[1])} faces.")

#         if len(mesh_data[0]) == 0 or len(mesh_data[1]) == 0:
#             print(f"Warning: No vertex or face data available for step {step}.")
#         else:
#             save_plot_as_html(contact_data, mesh_data, f'visualization_step_{step}', output_folder)
#         if os.path.exists(object_mesh_file_path):
#             object_mesh_data = load_obj_data(object_mesh_file_path)
#             save_plot_as_html(contact_data, mesh_data, object_mesh_data, f'visualization_step_{step}', output_folder)
#         else:
#             print(f"Warning: No object mesh file found for step {step}.")
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

    # 从文件名中提取步骤编号
    step = contact_file.split('_')[-1].split('.')[0]
    
    # 构建手部网格文件和物体网格文件的路径
    mesh_file = f'hand_mesh_step_{step}.obj'
    object_mesh_file = f'object_mesh_step_{step}_0.obj'  # 假设每一步都有一个名为 object_mesh_step_{step}_0.obj 的物体网格文件
    
    # 加载并保存网格数据和接触点数据
    if mesh_file in mesh_files:
        mesh_file_path = os.path.join(hand_mesh_folder, mesh_file)
        hand_mesh_data = load_obj_data(mesh_file_path)

        object_mesh_file_path = os.path.join(optimize_process_obj_object_directory, object_mesh_file)
        if os.path.exists(object_mesh_file_path):
            object_mesh_data = load_obj_data(object_mesh_file_path)
            save_plot_as_html(contact_data, hand_mesh_data, object_mesh_data, f'visualization_step_{step}', output_folder)
        else:
            print(f"Warning: No object mesh file found for step {step}.")
    else:
        print(f"Warning: No hand mesh file found for step {step}.")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import json
import numpy as np
import os

def load_contact_points(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def update_plot(num, contact_files, hand_mesh_files, object_mesh_files, ax):
    ax.clear()
    contact_data = load_contact_points(contact_files[num])
    hand_mesh_data = load_obj_data(hand_mesh_files[num])
    object_mesh_data = load_obj_data(object_mesh_files[num])

    # 绘制接触点
    for point in contact_data:
        ax.scatter(*point, color='blue')

    # 绘制手部mesh
    # ...

    # 绘制物体mesh
    # ...

# 文件夹路径和文件列表
contact_files = sorted([os.path.join(optimize_process_json_directory, f) for f in os.listdir(optimize_process_json_directory) if f.endswith('.json')])
hand_mesh_files = sorted([os.path.join(optimize_process_obj_hand_directory, f) for f in os.listdir(optimize_process_obj_hand_directory) if f.endswith('.obj')])
object_mesh_files = sorted([os.path.join(optimize_process_obj_object_directory, f) for f in os.listdir(optimize_process_obj_object_directory) if f.endswith('.obj')])

# 创建绘图和动画
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, update_plot, frames=len(contact_files), fargs=(contact_files, hand_mesh_files, object_mesh_files, ax), interval=200)

# 保存为MP4
ani.save('grasp_generation.mp4', writer='ffmpeg', fps=5)

plt.show()