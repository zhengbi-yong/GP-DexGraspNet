"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: visualize hand model
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath("."))

import json

import numpy as np
import plotly.graph_objects as go
import torch
import transforms3d
import trimesh as tm
from utils.leaphand_model import LEAPHandModel
from utils.config import get_abspath
torch.manual_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
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
def save_data_to_json(data, file_path):
    # 将张量转换为numpy数组，然后转换为列表
    data_list = data.detach().cpu().numpy().tolist()

    # 保存为JSON文件
    with open(file_path, 'w') as file:
        json.dump(data_list, file)
        
def save_leaphand_model_attributes(model, folder_path):
    # 创建文件夹，如果不存在
    os.makedirs(folder_path, exist_ok=True)

    for attr_name in dir(model):
        # 排除内置属性和方法
        if attr_name.startswith("__") or callable(getattr(model, attr_name)):
            continue

        # 获取属性值
        attr_value = getattr(model, attr_name)

        # 保存路径
        file_path = os.path.join(folder_path, attr_name)

        # 判断属性类型并选择保存方法
        if isinstance(attr_value, torch.Tensor):
            # 保存PyTorch张量
            torch.save(attr_value, file_path + ".pt")
        elif isinstance(attr_value, (dict, list)):
            # 保存为JSON格式
            with open(file_path + ".json", "w") as file:
                json.dump(attr_value, file, default=str)  # 使用str处理无法直接序列化的数据
        else:
            # 保存基本数据类型为文本文件
            with open(file_path + ".txt", "w") as file:
                file.write(str(attr_value))


if __name__ == "__main__":
    device = torch.device("cpu")

    # hand model
    hand_model = LEAPHandModel(
        urdf_path="/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand_centered/leaphand_right.urdf",
        contact_points_path="/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand_centered/local_contact_points.json",
        n_surface_points=1000,
        device=device,
    )

    # rot = transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes='rzyz')
    rot = transforms3d.euler.euler2mat(0, 0, 0, axes="rzyz")
    # hand_pose = torch.cat([
    #     torch.tensor([0, 0, 0], dtype=torch.float, device=device),
    #     # torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=device),
    #     torch.tensor(rot.T.ravel()[:6], dtype=torch.float, device=device),
    #     # torch.zeros([16], dtype=torch.float, device=device),
    #     torch.tensor([
    #         0, 0.5, 0, 0,
    #         0, 0.5, 0, 0,
    #         0, 0.5, 0, 0,
    #         1.4, 0, 0, 0,
    #     ], dtype=torch.float, device=device),
    # ], dim=0)
    hand_pose = torch.cat(
        [
            torch.tensor([0, 0, 0], dtype=torch.float, device=device),
            # torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=device),
            torch.tensor(rot.T.ravel()[:6], dtype=torch.float, device=device),
            # torch.zeros([16], dtype=torch.float, device=device),
            torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=torch.float,
                device=device,
            ),
        ],
        dim=0,
    )
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # info
    # contact_candidates = hand_model.get_contact_candidates()[0]
    contact_candidates = hand_model.get_contact_candidates_unchange()[0]
    surface_points = hand_model.get_surface_points()[0]
    save_data_to_json(contact_candidates, 'debug_contact_candidates.json')
    save_data_to_json(surface_points, 'debug_surface_points.json')
    hand_model.get_surface_points_local_and_save()
    print(f"n_dofs: {hand_model.n_dofs}")
    print(f"n_contact_candidates: {len(contact_candidates)}")
    print(f"n_surface_points: {len(surface_points)}")
    # print(f"surface_points_dict: {len(surface_points_dict)}")
    print(hand_model.chain.get_joint_parameter_names())

    hand_plotly = hand_model.get_plotly_data(
        i=0, opacity=0.3, color="lightblue", visual=False
    )
    v = contact_candidates.detach().cpu()
    contact_candidates_plotly = [
        go.Scatter3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            mode="markers",
            marker=dict(size=2, color="red"),
        )
    ]
    v = surface_points.detach().cpu()
    surface_points_plotly = [
        go.Scatter3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            mode="markers",
            marker=dict(size=2, color="green"),
        )
    ]

    # save_leaphand_model_attributes(hand_model, '/home/sisyphus/Allegro/DexGraspNet/note/LEAPHandModel')

    # fig = go.Figure(hand_plotly + contact_candidates_plotly + surface_points_plotly)
    fig = go.Figure(hand_plotly + contact_candidates_plotly)
    fig.update_layout(scene_aspectmode="data")
    fig.write_html(f"{hand_model_directory}/leaphand_centered_model.html")
    
    # 画一幅新的手
    # 应用新的手部姿态
    new_hand_pose = torch.cat(
    [
        torch.tensor([0, 0, 0], dtype=torch.float, device=device),
        torch.tensor(rot.T.ravel()[:6], dtype=torch.float, device=device),
        torch.tensor(
            [
                0.1, 0.3, 0, 0,  # 指关节的一些调整
                0, 0.4, 0, 0,
                0, 0.3, 0, 0,
                1.4, 0, 0, 0,
            ],
            dtype=torch.float,
            device=device,
        ),
    ],
    dim=0,
    )
    hand_model.set_parameters(new_hand_pose.unsqueeze(0))

    # 重新计算接触点的位置
    contact_candidates = hand_model.get_contact_candidates()[0]  # 或 get_contact_candidates_unchange()

    # 生成新的可视化数据
    # hand_plotly = hand_model.get_plotly_data(
    #     i=0, opacity=0.3, color="lightblue", visual=False
    # )
    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.3, color="lightblue", visual=True,collision=True)

    # 更新接触点的可视化
    contact_candidates_plotly = [
        go.Scatter3d(
            x=contact_candidates[:, 0].detach().cpu(),
            y=contact_candidates[:, 1].detach().cpu(),
            z=contact_candidates[:, 2].detach().cpu(),
            mode="markers",
            marker=dict(size=2, color="red"),
        )
    ]

    # 保存新的可视化文件
    fig = go.Figure(hand_plotly + contact_candidates_plotly)
    fig.update_layout(scene_aspectmode="data")
    fig.write_html(f"{hand_model_directory}/leaphand_centered_model_new_pose.html")