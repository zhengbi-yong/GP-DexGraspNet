"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: energy functions
"""

import torch
import json
"""
计算接触距离能量(E_dis):

计算手部接触点与对象表面之间的距离，并将这些距离的绝对值求和。
计算摩擦约束能量(E_fc):

使用接触点和接触法线计算摩擦约束，这与手部接触点是否滑动有关。
计算关节限制能量(E_joints):

检查手部的关节角度是否超出预设的上下限，并对超出的部分计算能量。
计算穿透能量(E_pen):

计算手部与对象表面的穿透距离，并对这些距离求和。
计算自我穿透能量(E_spen):

计算手部自身部件之间的穿透程度。
输出:

根据是否开启verbose模式，函数返回总能量和各种类型能量的详细值或只返回总能量。
此函数主要用于在机器学习模型中评估特定抓取姿势的有效性，通过计算不同类型的能量来判断抓取的可行性和稳定性。
"""


def cal_energy(
    hand_model,
    object_model,
    w_dis=100.0,
    w_pen=100.0,
    w_spen=10.0,
    w_joints=1.0,
    verbose=False,
):
    # hand_model.contact_points: 尺寸为 [batch_size, n_contact, 3]。这是手模型中的接触点坐标，batch_size 是抓取的数量，n_contact 是每个抓取的接触点数，3 表示空间坐标。
    # E_dis
    batch_size, n_contact, _ = hand_model.contact_points.shape
    def save_data_to_json(data, file_path):
        # 将张量转换为numpy数组，然后转换为列表
        data_list = data.detach().cpu().numpy().tolist()

        # 保存为JSON文件
        with open(file_path, 'w') as file:
            json.dump(data_list, file)
    save_data_to_json(hand_model.contact_points, 'debug_hand_model_contact_points.json')
    device = object_model.device
    # distance: 尺寸为 [batch_size, n_contact]。这是手部接触点与对象表面点之间的距离。
    # contact_normal: 尺寸为 [batch_size, n_contact, 3]。接触点的法线向量。
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    # E_dis: 尺寸为 [batch_size]。接触点距离的绝对值之和，表示接触距离能量。
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc
    # E_fc: 尺寸为 [batch_size]。基于接触法线和手部接触点的摩擦约束能量。
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor(
        [
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float,
        device=device,
    )
    g = (
        torch.cat(
            [
                torch.eye(3, dtype=torch.float, device=device)
                .expand(batch_size, n_contact, 3, 3)
                .reshape(batch_size, 3 * n_contact, 3),
                (hand_model.contact_points @ transformation_matrix).view(
                    batch_size, 3 * n_contact, 3
                ),
            ],
            dim=2,
        )
        .float()
        .to(device)
    )
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm

    # E_joints
    # E_joints: 尺寸为 [batch_size]。基于手部关节角度的关节限制能量。
    E_joints = torch.sum(
        (hand_model.hand_pose[:, 9:] > hand_model.joints_upper)
        * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper),
        dim=-1,
    ) + torch.sum(
        (hand_model.hand_pose[:, 9:] < hand_model.joints_lower)
        * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]),
        dim=-1,
    )

    # E_pen
    # E_pen: 尺寸为 [batch_size]。穿透能量，即手部穿透到对象内部的程度。
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    # object_surface_points: 尺寸为 [n_objects * batch_size_each, num_samples, 3]。对象表面的采样点。
    object_surface_points = (
        object_model.surface_points_tensor * object_scale
    )  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    # E_spen
    # E_spen: 尺寸为 [batch_size]。自我穿透能量，即手部自身部分相互穿透的程度。
    E_spen = hand_model.self_penetration()

    if verbose:
        return (
            E_fc
            + w_dis * E_dis
            + w_pen * E_pen
            + w_spen * E_spen
            + w_joints * E_joints,
            E_fc,
            E_dis,
            E_pen,
            E_spen,
            E_joints,
        )
    else:
        return (
            E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints
        )