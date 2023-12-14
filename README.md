# GP-DexGraspNet
Generating dexterous grasp poses.

Reference repository of [DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation](https://arxiv.org/abs/2210.02697).

[[project page]](https://pku-epic.github.io/DexGraspNet/)

## Environment Setup

```bash
conda create -n dexgraspnet python=3.7
conda activate dexgraspnet

# for quick example, cpu version is OK.
conda install pytorch cpuonly -c pytorch
conda install ipykernel
conda install transforms3d
conda install trimesh
pip install pyyaml
pip install lxml

cd thirdparty/pytorch_kinematics
pip install -e .
```
