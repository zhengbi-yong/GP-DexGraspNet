"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: generate grasps in large-scale, use multiple graphics cards, no logging
"""

import os
import sys
import json
# 将当前目录添加到系统路径，确保可以导入当前目录下的模块。
sys.path.append(os.path.realpath("."))

import argparse
import math
import multiprocessing
import random

import numpy as np
import torch
import transforms3d
from torch.multiprocessing import set_start_method
from tqdm import tqdm
from utils.energy import cal_energy
from utils.initializations import initialize_convex_hull
from utils.leaphand_model import LEAPHandModel
from utils.object_model import ObjectModel
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.config import get_abspath
import plotly.graph_objects as go

try:
    set_start_method("spawn")
except RuntimeError:
    pass

# 允许重复的库存在，这在使用某些特定的库时可能需要。
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# 设置NumPy在遇到数学错误时抛出异常，而不是默认的警告。
np.seterr(all="raise")

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

# 定义一些debug和展示优化过程的函数
def save_mesh_as_obj(mesh_data, file_path):
    with open(file_path, 'w') as f:
        for v in mesh_data['vertices']:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in mesh_data['faces']:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_contact_points_as_json(hand_model, step, save_dir):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将张量转换为numpy数组，然后转换为列表
    contact_points = hand_model.contact_points.detach().cpu().numpy().tolist()

    # 构建文件名并保存为JSON文件
    file_name = f"contact_points_step_{step}.json"
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, 'w') as file:
        json.dump(contact_points, file)


# 代码的核心，生成抓握姿势
def generate(args_list):
    # 将传入的args_list解包到四个变量中：args（实验参数），object_code_list（要处理的对象列表），id（进程ID），gpu_list（GPU列表）。
    args, object_code_list, id, gpu_list = args_list

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare models

    n_objects = len(object_code_list)

    worker = multiprocessing.current_process()._identity[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[worker - 1]
    device = torch.device("cuda")

    # 初始化手模型(LEAPHandModel)和对象模型(ObjectModel)。
    hand_model = LEAPHandModel(
        urdf_path=f"{leaphandcentered_directory}/leaphand_right.urdf",
        contact_points_path=f"{leaphandcentered_directory}/contact_points.json",
        n_surface_points=1000,
        device=device,
    )

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=2000,
        device=device,
    )
    object_model.initialize(object_code_list)

    # 初始化凸包，这是为了在模拟中使用凸包代表物体。
    initialize_convex_hull(hand_model, object_model, args)
    # debug: 查看初始化状态
    hand_st_plotly = hand_model.get_plotly_data(
        i=0, opacity=1, color="lightblue", with_contact_points=False, visual=False
    )
    # print(f"hand_pose_st: {hand_pose_st}")
    object_plotly = object_model.get_plotly_data(i=0, color="lightgreen", opacity=1)
    fig = go.Figure(hand_st_plotly + object_plotly)
    fig.update_layout(scene_aspectmode="data")
    fig.write_html("init0.html")

    hand_pose_st = hand_model.hand_pose.detach()
    # print(f"hand_pose_st[0]: {hand_pose_st[0]}")
    # print(f"hand_pose_st[0].shape: {hand_pose_st[0].shape}")
    optim_config = {
        "switch_possibility": args.switch_possibility,
        "starting_temperature": args.starting_temperature,
        "temperature_decay": args.temperature_decay,
        "annealing_period": args.annealing_period,
        "step_size": args.step_size,
        "stepsize_period": args.stepsize_period,
        "mu": args.mu,
        "device": device,
    }
    optimizer = Annealing(hand_model, **optim_config)

    # optimize

    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_spen=args.w_spen,
        w_joints=args.w_joints,
    )
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(
        hand_model, object_model, verbose=True, **weight_dict
    )

    energy.sum().backward(retain_graph=True)
    # debug: 查看初始化状态
    hand_st_plotly = hand_model.get_plotly_data(
        i=0, opacity=1, color="lightblue", with_contact_points=False, visual=False
    )
    # print(f"hand_pose_st: {hand_pose_st}")
    object_plotly = object_model.get_plotly_data(i=0, color="lightgreen", opacity=1)
    fig = go.Figure(hand_st_plotly + object_plotly)
    fig.update_layout(scene_aspectmode="data")
    fig.write_html("init.html")
    
    
    # for step in range(0, args.n_iter + 1):
    for step in tqdm(range(0, args.n_iter + 1), desc="Generating", unit="step"):
        if step % 500 == 0:
            # 保存手部网格数据和接触点数据，仅针对第一个物体
            if id == 1:  # 检查是否是第一个物体
                hand_mesh_data = hand_model.get_collision_mesh_data()
                save_mesh_as_obj(hand_mesh_data, os.path.join(optimize_process_obj_hand_directory, f"hand_mesh_step_{step}.obj"))
                
                save_contact_points_as_json(hand_model, step, optimize_process_json_directory)

            # 获取并保存物体网格数据
            if id == 1:  # 仅保存第一个物体的网格数据
                object_mesh_data = object_model.get_mesh_data()
                for i, mesh_data in enumerate(object_mesh_data):
                    if i == 0:  # 仅处理第一个物体
                        save_mesh_as_obj(mesh_data, os.path.join(optimize_process_obj_object_directory, f"object_mesh_step_{step}_{i}.obj"))
            
        s = optimizer.try_step()

        optimizer.zero_grad()
        (
            new_energy,
            new_E_fc,
            new_E_dis,
            new_E_pen,
            new_E_spen,
            new_E_joints,
        ) = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]

    # save results
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
    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(args.batch_size_each):
            idx = i * args.batch_size_each + j
            scale = object_model.object_scale_tensor[i][j].item()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
            rot = robust_compute_rotation_matrix_from_ortho6d(
                hand_pose[3:9].unsqueeze(0)
            )[0]
            euler = transforms3d.euler.mat2euler(rot, axes="sxyz")
            qpos.update(dict(zip(rot_names, euler)))
            qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            hand_pose = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
            rot = robust_compute_rotation_matrix_from_ortho6d(
                hand_pose[3:9].unsqueeze(0)
            )[0]
            euler = transforms3d.euler.mat2euler(rot, axes="sxyz")
            qpos_st.update(dict(zip(rot_names, euler)))
            qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            data_list.append(
                dict(
                    scale=scale,
                    qpos=qpos,
                    qpos_st=qpos_st,
                    energy=energy[idx].item(),
                    E_fc=E_fc[idx].item(),
                    E_dis=E_dis[idx].item(),
                    E_pen=E_pen[idx].item(),
                    E_spen=E_spen[idx].item(),
                    E_joints=E_joints[idx].item(),
                )
            )
        np.save(
            os.path.join(args.result_path, object_code + ".npy"),
            data_list,
            allow_pickle=True,
        )


if __name__ == "__main__":
    # 处理命令行输入的参数
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--result_path", default="../data/graspdata", type=str)
    parser.add_argument("--data_root_path", default="../data/meshdata", type=str)
    parser.add_argument("--object_code_list", nargs="*", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--todo", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_contact", default=4, type=int)
    parser.add_argument("--batch_size_each", default=10, type=int)
    parser.add_argument("--max_total_batch_size", default=100, type=int)
    parser.add_argument("--n_iter", default=6000, type=int)
    # hyper parameters
    parser.add_argument("--switch_possibility", default=0.5, type=float)
    parser.add_argument("--mu", default=0.98, type=float)
    parser.add_argument("--step_size", default=0.005, type=float)
    parser.add_argument("--stepsize_period", default=50, type=int)
    parser.add_argument("--starting_temperature", default=18, type=float)
    parser.add_argument("--annealing_period", default=30, type=int)
    parser.add_argument("--temperature_decay", default=0.95, type=float)
    parser.add_argument("--w_dis", default=100.0, type=float)
    parser.add_argument("--w_pen", default=100.0, type=float)
    parser.add_argument("--w_spen", default=10.0, type=float)
    parser.add_argument("--w_joints", default=1.0, type=float)
    # initialization settings
    parser.add_argument("--jitter_strength", default=0.1, type=float)
    parser.add_argument("--distance_lower", default=0.2, type=float)
    parser.add_argument("--distance_upper", default=0.3, type=float)
    parser.add_argument("--theta_lower", default=-math.pi / 6, type=float)
    parser.add_argument("--theta_upper", default=math.pi / 6, type=float)
    # energy thresholds
    parser.add_argument("--thres_fc", default=0.3, type=float)
    parser.add_argument("--thres_dis", default=0.005, type=float)
    parser.add_argument("--thres_pen", default=0.001, type=float)

    args = parser.parse_args()

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"gpu_list: {gpu_list}")

    # check whether arguments are valid and process arguments

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.exists(args.data_root_path):
        raise ValueError(f"data_root_path {args.data_root_path} doesn't exist")

    if (args.object_code_list is not None) + args.all != 1:
        raise ValueError(
            "exactly one among 'object_code_list' 'all' should be specified"
        )

    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = os.listdir(args.data_root_path)

    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError(
                "object_code_list isn't a subset of dirs in data_root_path"
            )
    else:
        object_code_list = object_code_list_all

    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, object_code + ".npy")):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(
            f"batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}"
        )

    print(f"number of objects: {len(object_code_list)}")

    # generate

    random.seed(args.seed)
    random.shuffle(object_code_list)
    objects_each = args.max_total_batch_size // args.batch_size_each
    object_code_groups = [
        object_code_list[i : i + objects_each]
        for i in range(0, len(object_code_list), objects_each)
    ]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    with multiprocessing.Pool(len(gpu_list)) as p:
        it = tqdm(
            p.imap(generate, process_args),
            total=len(process_args),
            desc="generating",
            maxinterval=1000,
        )
        list(it)