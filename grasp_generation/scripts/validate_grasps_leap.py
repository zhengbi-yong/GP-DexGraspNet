"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))
from utils.isaac_validator import IsaacValidator
import argparse
import numpy as np
import transforms3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.leaphand_model import LEAPHandModel
import torch
"""
这段代码是用于在Isaac模拟器上验证抓取动作的脚本。它主要执行以下几个步骤：加载抓取数据，使用Isaac模拟器运行抓取动作，并根据模拟结果来评估抓取的有效性。以下是对代码主要部分的解释以及数据结构的尺寸和含义：
参数解析：使用argparse库来解析命令行参数，这些参数包括GPU编号、批处理大小、路径信息和对象代码。

环境设置：设置CUDA环境变量，确保使用指定的GPU。

加载数据：从args.grasp_path中加载抓取数据，该数据包括每个抓取的姿势和相关信息。

处理手部状态：如果--no_force未启用，则调整手部状态以减少穿透。

设置模拟器：初始化Isaac模拟器，并根据是否指定了索引来设置为GUI模式或非GUI模式。

运行模拟：在模拟器中加载手和对象模型，运行模拟以评估抓取的有效性。

收集结果：收集模拟结果，并根据模拟和估算的穿透阈值来判断哪些抓取是有效的。

保存结果：将有效的抓取数据保存到指定的路径。

这个脚本的主要目的是通过在物理模拟器中运行抓取动作来验证这些动作的实际效果，从而评估抓取的质量。这种方法能够在实际部署到机器人之前提供对抓取动作有效性的预先了解。
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--val_batch", default=500, type=int)
    parser.add_argument("--mesh_path", default="../data/meshdata", type=str)
    parser.add_argument("--grasp_path", default="../data/leaphand_graspdata_version1_debug02", type=str)
    parser.add_argument("--result_path", default="../data/leaphand_graspdata_version1_debug02_result", type=str)
    parser.add_argument(
        "--object_code", default="core-cellphone-53ce93c96d480cc4da0f54fde38627c3", type=str
    )
    # if index is received, then the debug mode is on
    parser.add_argument("--index", type=int)
    parser.add_argument("--no_force", action="store_true")
    parser.add_argument("--thres_cont", default=0.001, type=float)
    parser.add_argument("--dis_move", default=0.001, type=float)
    parser.add_argument("--grad_move", default=500, type=float)
    parser.add_argument("--penetration_threshold", default=0.001, type=float)
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode in Isaac simulator",default=True)

    args = parser.parse_args()

    translation_names = ["WRJTx", "WRJTy", "WRJTz"]
    rot_names = ["WRJRx", "WRJRy", "WRJRz"]
    joint_names = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]

    os.environ.pop("CUDA_VISIBLE_DEVICES")
    os.makedirs(args.result_path, exist_ok=True)

    if not args.no_force:
        device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        data_dict = np.load(
            os.path.join(args.grasp_path, args.object_code + ".npy"), allow_pickle=True
        )
        batch_size = data_dict.shape[0]
        # hand_state：尺寸不直接给出，但是基于data_dict[i]['qpos']生成，是一个包含抓取姿势的张量。hand_state包含每个抓取的位移、旋转和关节角度。
        hand_state = []
        # scale_tensor：尺寸为[1, batch_size]，保存了每个抓取对象的缩放因子。
        scale_tensor = []
        for i in range(batch_size):
            print(f"data_dict[i]:{data_dict[i]}")
            qpos = data_dict[i]["qpos"]
            scale = data_dict[i]["scale"]
            rot = np.array(
                transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names])
            )
            rot = rot[:, :2].T.ravel().tolist()
            hand_pose = torch.tensor(
                [qpos[name] for name in translation_names]
                + rot
                + [qpos[name] for name in joint_names],
                dtype=torch.float,
                device=device,
            )
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)
        # print(scale_tensor.dtype)
        # hand_model = HandModel(
        #     mjcf_path="mjcf/shadow_hand_wrist_free.xml",
        #     mesh_path="mjcf/meshes",
        #     contact_points_path="mjcf/contact_points.json",
        #     penetration_points_path="mjcf/penetration_points.json",
        #     n_surface_points=2000,
        #     device=device,
        # )
        hand_model = LEAPHandModel(
        urdf_path="/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand_centered/leaphand_right.urdf",
        # urdf_path="/home/sisyphus/Allegro/DexGraspNet/grasp_generation/leaphand/leaphand_right.urdf",
        contact_points_path="/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand_centered/contact_points.json",
        n_surface_points=1000,
        device=device,
        )
        hand_model.set_parameters(hand_state)
        # object model
        object_model = ObjectModel(
            data_root_path=args.mesh_path,
            batch_size_each=batch_size,
            num_samples=0,
            device=device,
        )
        object_model.initialize(args.object_code)
        object_model.object_scale_tensor = scale_tensor

        # calculate contact points and contact normals
        # contact_points_hand 和 contact_normals：尺寸为[batch_size, 19, 3]，分别表示接触点的坐标和对应的法线向量。
        contact_points_hand = torch.zeros((batch_size, 19, 3)).to(device)
        contact_normals = torch.zeros((batch_size, 19, 3)).to(device)

        for i, link_name in enumerate(hand_model.mesh):
            if len(hand_model.mesh[link_name]["surface_points"]) == 0:
                continue
            surface_points = (
                hand_model.current_status[link_name]
                .transform_points(hand_model.mesh[link_name]["surface_points"])
                .expand(batch_size, -1, 3)
            )
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2
            ) + hand_model.global_translation.unsqueeze(1)
            distances, normals = object_model.cal_distance(surface_points)
            nearest_point_index = distances.argmax(dim=1)
            nearest_distances = torch.gather(
                distances, 1, nearest_point_index.unsqueeze(1)
            )
            nearest_points_hand = torch.gather(
                surface_points,
                1,
                nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
            )
            nearest_normals = torch.gather(
                normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
            )
            admited = -nearest_distances < args.thres_cont
            admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
            contact_points_hand[:, i : i + 1, :] = torch.where(
                admited, nearest_points_hand, contact_points_hand[:, i : i + 1, :]
            )
            contact_normals[:, i : i + 1, :] = torch.where(
                admited, nearest_normals, contact_normals[:, i : i + 1, :]
            )

        # target_points：尺寸为[batch_size, 19, 3]，表示调整后的接触点位置。
        target_points = contact_points_hand + contact_normals * args.dis_move
        loss = (target_points.detach().clone() - contact_points_hand).square().sum()
        loss.backward()
        with torch.no_grad():
            hand_state[:, 9:] += hand_state.grad[:, 9:] * args.grad_move
            hand_state.grad.zero_()

    # sim = IsaacValidator(gpu=args.gpu)
    # 使用新的 GUI 参数来决定是否使用 GUI 模式
    if args.gui:
        sim = IsaacValidator(gpu=args.gpu, mode="gui")
    else:
        sim = IsaacValidator(gpu=args.gpu)
    if args.index is not None:
        sim = IsaacValidator(gpu=args.gpu, mode="gui")

    data_dict = np.load(
        os.path.join(args.grasp_path, args.object_code + ".npy"), allow_pickle=True
    )
    batch_size = data_dict.shape[0]
    # rotations, translations, hand_poses 和 scale_array：这些数组包含了抓取动作的旋转、位移、手部姿势和缩放因子，用于在Isaac模拟器中设置抓取环境。
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    # E_pen_array：尺寸为[batch_size]，包含每个抓取的穿透能量值。
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]["qpos"]
        print(f"qpos:{qpos}")
        scale = data_dict[i]["scale"]
        rot = [qpos[name] for name in rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        translations.append(np.array([qpos[name] for name in translation_names]))
        hand_poses.append(np.array([qpos[name] for name in joint_names]))
        scale_array.append(scale)
        E_pen_array.append(data_dict[i]["E_pen"])
    E_pen_array = np.array(E_pen_array)
    if not args.no_force:
        hand_poses = hand_state[:, 9:]

    if args.index is not None:
        sim.set_asset(
            "open_ai_assets",
            "hand/shadow_hand.xml",
            os.path.join(args.mesh_path, args.object_code, "coacd"),
            "coacd.urdf",
        )
        index = args.index
        sim.add_env_single(
            rotations[index],
            translations[index],
            hand_poses[index],
            scale_array[index],
            0,
        )
        result = sim.run_sim()
        print(result)
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            sim.set_asset(
                "open_ai_assets",
                "hand/shadow_hand.xml",
                os.path.join(args.mesh_path, args.object_code, "coacd"),
                "coacd.urdf",
            )
            for index in range(offset, offset_):
                sim.add_env(
                    rotations[index],
                    translations[index],
                    hand_poses[index],
                    scale_array[index],
                )
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            offset = offset_
        for i in range(batch_size):
            simulated[i] = np.array(sum(result[i * 6 : (i + 1) * 6]) == 6)

        estimated = E_pen_array < args.penetration_threshold
        valid = simulated * estimated
        print(
            f"estimated: {estimated.sum().item()}/{batch_size}, "
            f"simulated: {simulated.sum().item()}/{batch_size}, "
            f"valid: {valid.sum().item()}/{batch_size}"
        )
        result_list = []
        for i in range(batch_size):
            if valid[i]:
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                result_list.append(new_data_dict)
        np.save(
            os.path.join(args.result_path, args.object_code + ".npy"),
            result_list,
            allow_pickle=True,
        )
    sim.destroy()