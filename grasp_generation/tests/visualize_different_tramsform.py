import json
import plotly.graph_objects as go
import numpy as np
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath("."))
from utils.config import get_abspath
# 获取一些需要使用的路径
scripts_directory = get_abspath(1,__file__)
graspgeneration_directory = get_abspath(2,__file__)
dexgraspnet_directory = get_abspath(3,__file__)
leaphandcentered_directory = os.path.join(graspgeneration_directory, "leaphand_centered")
optimize_process_directory = os.path.join(graspgeneration_directory, "debug/optimize_process")
optimize_process_json_directory = os.path.join(optimize_process_directory, "jsondata")
optimize_process_obj_directory = os.path.join(optimize_process_directory, "objdata")
optimize_process_obj_hand_directory = os.path.join(optimize_process_obj_directory, "hand")
optimize_process_obj_object_directory = os.path.join(optimize_process_obj_directory, "object")
debug_directory = os.path.join(graspgeneration_directory, "debug")
hand_model_directory = os.path.join(debug_directory, "handmodel")
# 假设您的数据文件名为 'file1.json' 和 'file2.json'
file1_path = f'{graspgeneration_directory}/surface_points.json'
file2_path = f'{graspgeneration_directory}/local_surface_points.json'

# 加载数据
with open(file1_path, 'r') as file:
    data1 = json.load(file)

with open(file2_path, 'r') as file:
    data2 = json.load(file)

# 为每个link生成一个独特的颜色
unique_links = list(set(data1.keys()) | set(data2.keys()))
colors = [f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 1)" for _ in unique_links]
link_to_color = dict(zip(unique_links, colors))

# 创建Plotly图表
fig = go.Figure()

# 添加 file1 数据
for link, points in data1.items():
    x, y, z = zip(*points)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(color=link_to_color[link], size=2),
        name=f'{link} (File 1)'
    ))

# 添加 file2 数据
for link, points in data2.items():
    x, y, z = zip(*points)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(color=link_to_color[link], size=2, symbol='circle'),
        name=f'{link} (File 2)'
    ))

# 更新图表布局
fig.update_layout(
    title='Link Points Visualization',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)
fig.write_html(f"{hand_model_directory}/transform_compare.html")
# 显示图表
# fig.show()