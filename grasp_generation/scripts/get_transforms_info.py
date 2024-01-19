# import xml.etree.ElementTree as ET
# import os
# import sys
# sys.path.append(os.path.realpath("."))
# from utils.config import get_abspath
# # 获取一些需要使用的路径
# scripts_directory = get_abspath(1,__file__)
# graspgeneration_directory = get_abspath(2,__file__)
# dexgraspnet_directory = get_abspath(3,__file__)
# leaphandcentered_directory = os.path.join(graspgeneration_directory, "leaphand_centered")
# optimize_process_directory = os.path.join(graspgeneration_directory, "debug/optimize_process")
# optimize_process_json_directory = os.path.join(optimize_process_directory, "jsondata")
# optimize_process_obj_directory = os.path.join(optimize_process_directory, "objdata")
# optimize_process_obj_hand_directory = os.path.join(optimize_process_obj_directory, "hand")
# optimize_process_obj_object_directory = os.path.join(optimize_process_obj_directory, "object")
# def parse_urdf_transforms(urdf_file_path):
#     """ 解析URDF文件以获取变换信息 """
#     transforms_info = {}

#     # 解析URDF文件
#     tree = ET.parse(urdf_file_path)
#     root = tree.getroot()

#     # 遍历所有link元素
#     for link in root.findall('link'):
#         link_name = link.get('name')
#         visual = link.find('visual')

#         if visual is not None:
#             origin = visual.find('origin')
#             if origin is not None:
#                 xyz = origin.get('xyz')
#                 rpy = origin.get('rpy')

#                 if xyz and rpy:
#                     # 转换xyz和rpy为浮点数列表
#                     xyz = [float(x) for x in xyz.split()]
#                     rpy = [float(x) for x in rpy.split()]

#                     transforms_info[link_name] = {"xyz": xyz, "rpy": rpy}

#     return transforms_info

# # 指定URDF文件的路径
# urdf_file_path = f'{leaphandcentered_directory}/leaphand_right.urdf'  # 替换为您的URDF文件路径

# # 解析URDF文件以获取变换信息
# transforms_info = parse_urdf_transforms(urdf_file_path)

# # 打印提取的信息
# for link, info in transforms_info.items():
#     print(f"Link: {link}, XYZ: {info['xyz']}, RPY: {info['rpy']}")
import json
import xml.etree.ElementTree as ET
import os
import sys
sys.path.append(os.path.realpath("."))
from utils.config import get_abspath

# 获取一些需要使用的路径
scripts_directory = get_abspath(1, __file__)
graspgeneration_directory = get_abspath(2, __file__)
dexgraspnet_directory = get_abspath(3, __file__)
leaphandcentered_directory = os.path.join(graspgeneration_directory, "leaphand_centered")
optimize_process_directory = os.path.join(graspgeneration_directory, "debug/optimize_process")
optimize_process_json_directory = os.path.join(optimize_process_directory, "jsondata")
optimize_process_obj_directory = os.path.join(optimize_process_directory, "objdata")
optimize_process_obj_hand_directory = os.path.join(optimize_process_obj_directory, "hand")
optimize_process_obj_object_directory = os.path.join(optimize_process_obj_directory, "object")

def parse_urdf_transforms(urdf_file_path):
    """ 解析URDF文件以获取变换信息 """
    transforms_info = {}

    # 解析URDF文件
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()

    # 遍历所有link元素
    for link in root.findall('link'):
        link_name = link.get('name')
        visual = link.find('visual')

        if visual is not None:
            origin = visual.find('origin')
            if origin is not None:
                xyz = origin.get('xyz')
                rpy = origin.get('rpy')

                if xyz and rpy:
                    # 转换xyz和rpy为浮点数列表
                    xyz = [float(x) for x in xyz.split()]
                    rpy = [float(x) for x in rpy.split()]

                    transforms_info[link_name] = {"xyz": xyz, "rpy": rpy}

    return transforms_info

# 指定URDF文件的路径
urdf_file_path = f'{leaphandcentered_directory}/leaphand_right.urdf'

# 解析URDF文件以获取变换信息
transforms_info = parse_urdf_transforms(urdf_file_path)

# 将提取的信息保存到文件
transforms_info_file = f'{leaphandcentered_directory}/transforms_info.json'
with open(transforms_info_file, 'w') as file:
    json.dump(transforms_info, file, indent=4)

print(f"Transforms info saved to {transforms_info_file}")
