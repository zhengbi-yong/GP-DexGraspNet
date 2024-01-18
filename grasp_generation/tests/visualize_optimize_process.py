import json
import plotly.graph_objects as go

# 加载 JSON 文件
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 创建 Plotly 图形对象
def create_plotly_scatter3d_3(data, marker_color='blue'):
    # 展开嵌套的列表结构
    flattened_data = [point for sublist in data for point in sublist]
    x, y, z = zip(*flattened_data)  # 提取 x, y, z 坐标
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=marker_color, size=5))
    return scatter

def create_plotly_scatter3d_2(data, marker_color='blue'):
    # 解析数据
    x, y, z = zip(*data)
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=marker_color, size=5))
    return scatter

# 文件路径
json_file_path1 = '/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/debug_before_update_hand_model_contact_points.json'
# json_file_path2 = '/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/debug_after_update_hand_model_contact_points.json'
json_file_path2 = '/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/debug_surface_points.json'
# 加载数据
json_data1 = load_json_data(json_file_path1)
json_data2 = load_json_data(json_file_path2)

# 创建散点图
scatter_plot1 = create_plotly_scatter3d_3(json_data1, marker_color='green')
scatter_plot2 = create_plotly_scatter3d_2(json_data2, marker_color='red')

# 创建图表并添加散点图
fig = go.Figure()
fig.add_trace(scatter_plot1)
fig.add_trace(scatter_plot2)

# 更新布局并显示图表
fig.update_layout(scene_aspectmode='data')
fig.write_html("debug_optimize_process.html")
# fig.show()