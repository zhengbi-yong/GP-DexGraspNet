import json
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)
    
def get_abspath(levels, caller_file):
    """
    返回指定级别上级目录的路径。
    
    参数:
    levels (int): 上级目录的级别。0 表示当前文件的目录，1 表示父级目录，依此类推。
    caller_file (str): 调用此函数的文件的 __file__ 属性。
    
    返回:
    str: 指定级别的目录路径。
    
    异常:
    ValueError: 如果指定的级别超过了目录树的根。
    """
    if levels < 0:
        raise ValueError("级别必须是非负整数")

    # 获取调用文件的完整路径
    path = os.path.abspath(caller_file)

    for _ in range(levels):
        new_path = os.path.dirname(path)
        if new_path == path:  # 已达到文件系统的根目录
            raise ValueError("已超过文件系统的根目录")
        path = new_path

    return path