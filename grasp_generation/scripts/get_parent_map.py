import xml.etree.ElementTree as ET
import json
import os

def generate_parent_map_from_urdf(urdf_file_path):
    parent_map = {}
    
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()

    for joint in root.findall('joint'):
        child_link = joint.find('child').get('link')
        parent_link = joint.find('parent').get('link')
        parent_map[child_link] = parent_link

    return parent_map

# 指定URDF文件的路径
urdf_file_path = '/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand_centered/leaphand_right.urdf'

# 生成parent_map
parent_map = generate_parent_map_from_urdf(urdf_file_path)

# 保存parent_map到JSON文件
parent_map_file = '/home/sisyphus/GP/GP-DexGraspNet/grasp_generation/leaphand_centered/parent_map.json'
with open(parent_map_file, 'w') as file:
    json.dump(parent_map, file, indent=4)

print(f"Parent map saved to {parent_map_file}")