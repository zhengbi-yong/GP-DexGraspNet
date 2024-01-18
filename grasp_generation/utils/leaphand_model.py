"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: Class HandModel
"""

import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import json

import numpy as np
import plotly.graph_objects as go
import pytorch3d.ops
import pytorch3d.structures
import pytorch_kinematics as pk
import torch
import transforms3d
import trimesh as tm
from torchsdf import compute_sdf, index_vertices_by_faces
from urdf_parser_py.urdf import Box, Mesh, Robot, Sphere
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


class LEAPHandModel:
    def __init__(
        self, urdf_path, contact_points_path, n_surface_points=0, device="cpu"
    ):
        self.device = device
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(
            dtype=torch.float, device=device
        )
        self.robot = Robot.from_xml_file(urdf_path)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        contact_points = json.load(open(contact_points_path, "r"))
        # def save_data_to_json(data, file_path):
        #     # 将张量转换为numpy数组，然后转换为列表
        #     data_list = data

        #     # 保存为JSON文件
        #     with open(file_path, 'w') as file:
        #         json.dump(data_list, file)
        # save_data_to_json(contact_points, 'debug_contact_points.json')
        self.mesh = {}
        areas = {}
        for link in self.robot.links:
            if link.visual is None or link.collision is None:
                continue
            self.mesh[link.name] = {}
            # load collision mesh
            collision = link.collision
            # print(f"collision.geometry:{collision.geometry}")
            if type(collision.geometry) == Sphere:
                link_mesh = tm.primitives.Sphere(radius=collision.geometry.radius)
                self.mesh[link.name]["radius"] = collision.geometry.radius
            if type(collision.geometry) == Box:
                # link_mesh = tm.primitives.Box(extents=collision.geometry.size)
                link_mesh = tm.load_mesh(
                    os.path.join(os.path.dirname(urdf_path), "meshes", "box.obj"),
                    process=False,
                )
                link_mesh.vertices *= np.array(collision.geometry.size) / 2
            if type(collision.geometry) == Mesh:
                # 打印以确保 collision.geometry.filename 包含正确的文件名
                # print(f"STL Filename: {collision.geometry.filename}")

                # 构建并打印完整的文件路径
                filename = os.path.join(
                    "/home/sisyphus/Allegro/DexGraspNet/grasp_generation/leaphand",
                    collision.geometry.filename,
                )
                # print(f"Full STL Path: {filename}")

                # 尝试加载文件
                try:
                    link_mesh = tm.load_mesh(filename)
                except ValueError as e:
                    print(f"Error loading STL file: {e}")

            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device
            )
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if (
                hasattr(collision.geometry, "scale")
                and collision.geometry.scale is None
            ):
                collision.geometry.scale = [1, 1, 1]
            scale = torch.tensor(
                getattr(collision.geometry, "scale", [1, 1, 1]),
                dtype=torch.float,
                device=device,
            )
            translation = torch.tensor(
                getattr(collision.origin, "xyz", [0, 0, 0]),
                dtype=torch.float,
                device=device,
            )
            rotation = torch.tensor(
                transforms3d.euler.euler2mat(
                    *getattr(collision.origin, "rpy", [0, 0, 0])
                ),
                dtype=torch.float,
                device=device,
            )
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update(
                {
                    "vertices": vertices,
                    "faces": faces,
                }
            )
            if "radius" not in self.mesh[link.name]:
                self.mesh[link.name]["face_verts"] = index_vertices_by_faces(
                    vertices, faces
                )
            areas[link.name] = tm.Trimesh(
                vertices.cpu().numpy(), faces.cpu().numpy()
            ).area.item()
            # load visual mesh
            visual = link.visual
            filename = os.path.join(
                os.path.dirname(urdf_path), visual.geometry.filename
            )
            link_mesh = tm.load_mesh(filename)
            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device
            )
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if hasattr(visual.geometry, "scale") and visual.geometry.scale is None:
                visual.geometry.scale = [1, 1, 1]
            scale = torch.tensor(
                getattr(visual.geometry, "scale", [1, 1, 1]),
                dtype=torch.float,
                device=device,
            )
            translation = torch.tensor(
                getattr(visual.origin, "xyz", [0, 0, 0]),
                dtype=torch.float,
                device=device,
            )
            rotation = torch.tensor(
                transforms3d.euler.euler2mat(*getattr(visual.origin, "rpy", [0, 0, 0])),
                dtype=torch.float,
                device=device,
            )
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update(
                {
                    "visual_vertices": vertices,
                    "visual_faces": faces,
                }
            )
            # load contact candidates and penetration keypoints
            contact_candidates = torch.tensor(
                contact_points[link.name], dtype=torch.float32, device=device
            ).reshape(-1, 3)
            self.mesh[link.name].update(
                {
                    "contact_candidates": contact_candidates,
                }
            )

        self.joints_lower = torch.tensor(
            [
                joint.limit.lower
                for joint in self.robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )
        self.joints_upper = torch.tensor(
            [
                joint.limit.upper
                for joint in self.robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )

        # sample surface points
        total_area = sum(areas.values())
        num_samples = dict(
            [
                (link_name, int(areas[link_name] / total_area * n_surface_points))
                for link_name in self.mesh
            ]
        )
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(
            num_samples.values()
        )
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=device
                ).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(
                self.mesh[link_name]["vertices"].unsqueeze(0),
                self.mesh[link_name]["faces"].unsqueeze(0),
            )
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=100 * num_samples[link_name]
            )
            surface_points = pytorch3d.ops.sample_farthest_points(
                dense_point_cloud, K=num_samples[link_name]
            )[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]["surface_points"] = surface_points

        self.link_name_to_link_index = dict(
            zip([link_name for link_name in self.mesh], range(len(self.mesh)))
        )
        self.surface_points_link_indices = torch.cat(
            [
                self.link_name_to_link_index[link_name]
                * torch.ones(
                    self.mesh[link_name]["surface_points"].shape[0],
                    dtype=torch.long,
                    device=device,
                )
                for link_name in self.mesh
            ]
        )

        self.contact_candidates = [
            self.mesh[link_name]["contact_candidates"] for link_name in self.mesh
        ]
        self.global_index_to_link_index = sum(
            [
                [i] * len(contact_candidates)
                for i, contact_candidates in enumerate(self.contact_candidates)
            ],
            [],
        )
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=device
        )
        self.n_contact_candidates = self.contact_candidates.shape[0]

        # build collision mask
        self.adjacency_mask = torch.zeros(
            [len(self.mesh), len(self.mesh)], dtype=torch.bool, device=device
        )
        for joint in self.robot.joints:
            parent_id = self.link_name_to_link_index[joint.parent]
            child_id = self.link_name_to_link_index[joint.child]
            self.adjacency_mask[parent_id, child_id] = True
            self.adjacency_mask[child_id, parent_id] = True
        # self.adjacency_mask[self.link_name_to_link_index['base_link'], self.link_name_to_link_index['link_13.0']] = True
        # self.adjacency_mask[self.link_name_to_link_index['link_13.0'], self.link_name_to_link_index['base_link']] = True
        self.adjacency_mask[
            self.link_name_to_link_index["palm_lower"],
            self.link_name_to_link_index["pip_4"],
        ] = True
        self.adjacency_mask[
            self.link_name_to_link_index["pip_4"],
            self.link_name_to_link_index["palm_lower"],
        ] = True

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None
        # print(f"self.device:{self.device}")
        # print(f"self.chain:{self.chain}")
        # print(f"self.robot:{self.robot}")
        # print(f"self.n_dofs:{self.n_dofs}")
        # print(f"self.mesh:{self.mesh}")
        # print(f"self.joints_lower:{self.joints_lower}")
        # print(f"self.joints_upper:{self.joints_upper}")
        # print(f"self.link_name_to_link_index:{self.link_name_to_link_index}")
        # print(f"self.surface_points_link_indices:{self.surface_points_link_indices}")
        # print(f"self.contact_candidates:{self.contact_candidates}")
        # print(f"self.global_index_to_link_index:{self.global_index_to_link_index}")
        # print(f"self.n_contact_candidates:{self.n_contact_candidates}")
        # print(f"self.adjacency_mask:{self.adjacency_mask}")
        # print(f"self.hand_pose:{self.hand_pose}")
        # print(f"self.contact_point_indices:{self.contact_point_indices}")
        # print(f"self.global_translation:{self.global_translation}")
        # print(f"self.global_rotation:{self.global_rotation}")
        # print(f"self.current_status:{self.current_status}")
        # print(f"self.contact_points:{self.contact_points}")

    def set_parameters(self, hand_pose, contact_point_indices=None):
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9]
        )
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(
                batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device
            )
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = (
                    self.current_status[link_name]
                    .get_matrix()
                    .unsqueeze(1)
                    .expand(batch_size, n_contact, 4, 4)
                )
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat(
                [
                    self.contact_points,
                    torch.ones(
                        batch_size, n_contact, 1, dtype=torch.float, device=self.device
                    ),
                ],
                dim=2,
            )
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[
                :, :, :3, 0
            ]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(
                1, 2
            ) + self.global_translation.unsqueeze(1)

    def cal_distance(self, x):
        # x: (total_batch_size, num_samples, 3)
        # 单独考虑每个link
        # 先把x变换到link的局部坐标系里面，得到x_local: (total_batch_size, num_samples, 3)
        # 然后计算dis，按照内外取符号，内部是正号
        # 最后的dis就是所有link的dis的最大值
        # 对于sphere的link，使用解析方法计算dis，否则用mesh的方法计算dis
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if "radius" not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]["face_verts"]
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                dis_local = self.mesh[link_name]["radius"] - x_local.norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def cal_self_distance(self):
        # get surface points
        x = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            x.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["surface_points"]
                )
            )
            if 1 < batch_size != x[-1].shape[0]:
                x[-1] = x[-1].expand(batch_size, n_surface_points, 3)
        x = torch.cat(x, dim=-2).to(
            self.device
        )  # (total_batch_size, n_surface_points, 3)
        if len(x.shape) == 2:
            x = x.expand(1, x.shape[0], x.shape[1])
        # cal distance
        dis = []
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            # if len(matrix) != len(x):
            #     matrix = matrix.expand(len(x), 4, 4)
            # x_local = transform_points_inverse(x, matrix)
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * n_surface_points, 3)
            if "radius" in self.mesh[link_name]:
                radius = self.mesh[link_name]["radius"]
                dis_local = (
                    radius - (x_local.square().sum(-1) + 1e-8).sqrt()
                )  # (total_batch_size * n_surface_points,)
            else:
                face_verts = self.mesh[link_name]["face_verts"]
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = (dis_local + 1e-8).sqrt()
                dis_local = dis_local * (-dis_signs)
            dis_local = dis_local.reshape(
                x.shape[0], x.shape[1]
            )  # (total_batch_size, n_surface_points)
            is_adjacent = self.adjacency_mask[
                self.link_name_to_link_index[link_name],
                self.surface_points_link_indices,
            ]  # (n_surface_points,)
            dis_local[
                :,
                is_adjacent
                | (
                    self.link_name_to_link_index[link_name]
                    == self.surface_points_link_indices
                ),
            ] = -float("inf")
            dis.append(dis_local)
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self):
        dis = self.cal_self_distance()
        dis[dis <= 0] = 0
        E_spen = dis.sum(-1)
        return E_spen

    def get_contact_candidates(self):
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["contact_candidates"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["contact_candidates"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points
    
    def get_contact_candidates_unchange(self):
        points = []
        batch_size = self.global_translation.shape[0]

        for link_name in self.mesh:
            contact_candidates = self.mesh[link_name]["contact_candidates"]

            # 确保contact_candidates是二维的
            if len(contact_candidates.shape) == 1:
                contact_candidates = contact_candidates.unsqueeze(0)

            # 扩展到批次大小
            contact_candidates = contact_candidates.expand(batch_size, -1, 3)

            points.append(contact_candidates)

        # 将所有link的接触候选点合并为一个张量
        points = torch.cat(points, dim=1)  # 注意这里的维度是1，合并在第二个维度
        return points


    def get_mesh_data(self):
        """获取整个手部的网格数据。"""
        all_vertices = []
        all_faces = []
        face_offset = 0

        for link_name in self.mesh:
            link_data = self.mesh[link_name]
            vertices = link_data["vertices"]
            faces = link_data["faces"] + face_offset
            face_offset += len(vertices)

            all_vertices.append(vertices)
            all_faces.append(faces)

        # 合并所有链接的顶点和面
        all_vertices = torch.cat(all_vertices, dim=0)
        all_faces = torch.cat(all_faces, dim=0)

        return {
            'vertices': all_vertices.cpu().numpy(),
            'faces': all_faces.cpu().numpy()
        }


    def get_surface_points(self):
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["surface_points"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points
    
    def get_surface_points_and_save(self):
        surface_points_dict = {}
        batch_size = self.global_translation.shape[0]
        
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            transformed_points = self.current_status[link_name].transform_points(
                self.mesh[link_name]["surface_points"]
            )

            if 1 < batch_size != transformed_points.shape[0]:
                transformed_points = transformed_points.expand(batch_size, n_surface_points, 3)

            global_points = transformed_points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
            surface_points_dict[link_name] = global_points[0].cpu().numpy().tolist()

        # 保存为 JSON 文件
        with open('surface_points.json', 'w') as file:
            json.dump(surface_points_dict, file)
        return surface_points_dict

        

    def get_plotly_data(
        self, i, opacity=0.5, color="lightblue", with_contact_points=False, visual=False
    ):
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]["visual_vertices" if visual else "vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = (
                self.mesh[link_name]["visual_faces" if visual else "faces"]
                .detach()
                .cpu()
            )
            data.append(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    text=[link_name] * len(v),
                    color=color,
                    opacity=opacity,
                    hovertemplate="%{text}",
                )
            )
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            data.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                )
            )
        return data