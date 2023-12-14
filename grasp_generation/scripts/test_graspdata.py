import numpy as np

# 替换此处的文件路径为你的.npy文件的实际路径
file_path = "/home/sisyphus/DexGraspNet/data/leaphand_graspdata_version1/core.npy"

# 加载.npy文件
data = np.load(file_path, allow_pickle=True)

# 打印数据
print(data)
