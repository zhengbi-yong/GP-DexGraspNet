import sys
import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.append(os.path.realpath("."))
from utils.config import get_abspath

scripts_directory = get_abspath(1, __file__)
graspgeneration_directory = get_abspath(2, __file__)
dexgraspnet_directory = get_abspath(3, __file__)
leaphandcentered_directory = os.path.join(graspgeneration_directory, "leaphand_centered")
debug_directory = os.path.join(graspgeneration_directory, "debug")
optimize_process_directory = os.path.join(graspgeneration_directory, "debug/optimize_process")
optimize_process_json_directory = os.path.join(optimize_process_directory, "jsondata")
optimize_process_obj_directory = os.path.join(optimize_process_directory, "objdata")
optimize_process_obj_hand_directory = os.path.join(optimize_process_obj_directory, "hand")
optimize_process_obj_object_directory = os.path.join(optimize_process_obj_directory, "object")

def load_obj(file_path):
    """从 OBJ 文件中读取顶点和面数据。"""
    vertices = []
    faces = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(coord) for coord in parts[1:4]])
                elif line.startswith('f'):
                    parts = line.strip().split()
                    face = [int(idx.split('/')[0]) for idx in parts[1:]]
                    faces.append([idx - 1 for idx in face])
        print(f"Successfully loaded {len(vertices)} vertices and {len(faces)} faces from {file_path}")
        return np.array(vertices), np.array(faces)
    except Exception as e:
        print(f"Error reading OBJ file: {e}")
        return np.array([]), np.array([])

def filter_invalid_faces(vertices, faces):
    """筛选掉包含无效顶点索引的面。"""
    valid_faces = []
    for face in faces:
        if np.all(face < len(vertices)):
            valid_faces.append(face)
        else:
            print(f"Skipping invalid face: {face}")
    return np.array(valid_faces)

def plot_mesh(vertices, faces, title="3D Object", output_html_file="output.html"):
    if vertices.size == 0 or faces.size == 0:
        print("No data to plot.")
        return

    x, y, z = vertices.T
    i, j, k = np.array(faces).T

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightblue', opacity=0.5)

    fig = go.Figure(data=[mesh])
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    
    if len(fig.data) == 0:
        print("No data in the Plotly figure.")
    else:
        fig.write_html(output_html_file)
        print(f"Visualization saved to {output_html_file}")

def plot_mesh_plotly(vertices, faces, title="3D Object", output_html_file="output.html"):
    if vertices.size == 0 or faces.size == 0:
        print("No data to plot.")
        return

    # 筛选掉无效的面
    faces = filter_invalid_faces(vertices, faces)

    x, y, z = vertices.T
    i, j, k = faces.T

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightblue', opacity=0.5)
    fig = go.Figure(data=[mesh])
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    
    if len(fig.data) == 0:
        print("No data in the Plotly figure.")
    else:
        fig.write_html(output_html_file)
        print(f"Visualization saved to {output_html_file}")


def print_sample_data(vertices, faces, sample_size=5):
    print("Sample vertices:")
    print(vertices[:sample_size])
    print("Sample faces:")
    print(faces[:sample_size])

def plot_mesh_matplotlib(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 检查并绘制每个面
    for face in faces:
        if np.any(face >= len(vertices)):
            print("Skipping invalid face:", face)
            continue

        points = vertices[face]
        poly = Poly3DCollection([points])
        poly.set_color('lightblue')
        ax.add_collection3d(poly)

    ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
    ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
    ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])

    plt.show()

file_path = f'{optimize_process_obj_hand_directory}/hand_mesh_step_6000.obj'

vertices, faces = load_obj(file_path)
plot_mesh_plotly(vertices, faces, title="OBJ File Visualization", output_html_file=os.path.join(debug_directory, "hand_mesh_step_6000_visualization.html"))
# plot_mesh_matplotlib(vertices, faces)