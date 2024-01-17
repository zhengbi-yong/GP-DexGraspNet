import numpy as np

# 替换此处的文件路径为你的.npy文件的实际路径
file_path = "/home/sisyphus/GP/GP-DexGraspNet/data/leaphand_graspdata_version1_debug01/core-bottle-6d3c20adcce2dd0f79b7b9ca8006b2fe.npy"

# 加载.npy文件
data = np.load(file_path, allow_pickle=True)

# 打印数据
# print(data)
print(data.shape)
print(data[0])