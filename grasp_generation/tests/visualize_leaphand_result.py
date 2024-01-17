"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

import numpy as np
import plotly.graph_objects as go
import torch
import transforms3d
from utils.hand_model import HandModel
from utils.leaphand_model import LEAPHandModel
from utils.object_model import ObjectModel

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_code", type=str, default="core-bottle-6d3c20adcce2dd0f79b7b9ca8006b2fe"
    )
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument(
        "--result_path",
        type=str,
        default="/home/sisyphus/GP/GP-DexGraspNet/data/leaphand_graspdata_version1_debug01/",
    )
    args = parser.parse_args()

    device = "cpu"

    # load results
    print(f"args.num: {args.num}")
    data_dict = np.load(
        os.path.join(args.result_path, args.object_code + ".npy"), allow_pickle=True
    )[args.num]
    qpos = data_dict["qpos"]
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor(
        [qpos[name] for name in translation_names]
        + rot
        + [qpos[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    if "qpos_st" in data_dict:
        qpos_st = data_dict["qpos_st"]
        rot = np.array(
            transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names])
        )
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor(
            [qpos_st[name] for name in translation_names]
            + rot
            + [qpos_st[name] for name in joint_names],
            dtype=torch.float,
            device=device,
        )

    # hand model
    # hand_model = HandModel(
    #     mjcf_path="/home/sisyphus/DexGraspNet/grasp_generation/mjcf/shadow_hand_wrist_free.xml",
    #     mesh_path="/home/sisyphus/DexGraspNet/grasp_generation/mjcf/meshes",
    #     contact_points_path="/home/sisyphus/DexGraspNet/grasp_generation/mjcf/contact_points.json",
    #     penetration_points_path="/home/sisyphus/DexGraspNet/grasp_generation/mjcf/penetration_points.json",
    #     device=device,
    # )
    hand_model = LEAPHandModel(
        urdf_path="/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand/leaphand_right.urdf",
        contact_points_path="/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand/contact_points.json",
        n_surface_points=1000,
        device=device,
    )

    # object model
    object_model = ObjectModel(
        data_root_path="/home/sisyphus/GP/GP-DexGraspNet/data/meshdata",
        batch_size_each=1,
        num_samples=2000,
        device=device,
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(
        data_dict["scale"], dtype=torch.float, device=device
    ).reshape(1, 1)

    # visualize

    if "qpos_st" in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(
            i=0, opacity=0.5, color="lightblue", with_contact_points=False, visual=False
        )
    else:
        hand_st_plotly = []
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(
        i=0, opacity=1, color="lightblue", with_contact_points=False, visual=False
    )
    object_plotly = object_model.get_plotly_data(i=0, color="lightgreen", opacity=1)
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    if "energy" in data_dict:
        energy = data_dict["energy"]
        E_fc = round(data_dict["E_fc"], 3)
        E_dis = round(data_dict["E_dis"], 5)
        E_pen = round(data_dict["E_pen"], 5)
        E_spen = round(data_dict["E_spen"], 5)
        E_joints = round(data_dict["E_joints"], 5)
        result = f"Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}"
        fig.add_annotation(text=result, x=0.5, y=0.1, xref="paper", yref="paper")
    fig.update_layout(scene_aspectmode="data")
    fig.write_html("output.html")
    fig.show()