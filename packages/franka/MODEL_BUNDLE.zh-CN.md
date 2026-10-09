# Franka Panda 正式模型包

[English](MODEL_BUNDLE.md) | [简体中文](MODEL_BUNDLE.zh-CN.md)

SDK 仓不分发大型 Robot 资产，也不会从 `.venv`、robosuite、Isaac Sim 或 ROS
安装目录搜索模型。发布流水线需要先在资产仓生成一个独立、可审查的模型包，再把
模型包根目录通过 `FRANKA_MODEL_ROOT` 交给专项测试；生产代码则直接传入该路径。

## 来源与许可证

模型只接受 Franka Robotics 官方仓库：

- `https://github.com/frankarobotics/franka_ros`：传统 Panda 描述的来源。
- `https://github.com/frankarobotics/franka_description`：当前官方模型仓。

两者均以 Apache-2.0 发布，并在仓库中提供 `LICENSE` 和 `NOTICE`。资产 MR 必须：

1. 固定一个 Tag 或 Commit，不能记录 `main`、`master` 或 `latest`。
2. 保存同一 revision 的完整 `LICENSE` 与 `NOTICE`，不能只在 MR 描述里放链接。
3. 记录生成 `panda_arm_hand.urdf` 的命令和输入；生成结果的根 Link 必须是
   `panda_link0`，手部末端 Frame 必须是 `panda_hand`。
4. 保留 URDF 引用的碰撞 Mesh。只有视觉 Mesh、没有碰撞 Mesh 的包不能通过
   Pinocchio 自碰撞门控。
5. 由资产负责人核对文件来源和可分发范围。程序校验只能防止漏文件或配错版本，
   不能代替许可证审查。

## 目录和 manifest

推荐目录如下；`package_roots` 也允许另一种自包含层级，但不能离开模型包根目录。

```text
franka-model-bundle/
├── robot-model.json
├── LICENSE
├── NOTICE
└── franka_description/
    ├── robots/panda_arm_hand.urdf
    └── meshes/...
```

`robot-model.json` 使用仓库内的 `model-manifest.example.json` 作为模板。校验器会
检查型号、URDF 文件名、package root、根坐标系、末端坐标系、Apache-2.0 标记、
LICENSE、NOTICE、官方来源和固定 revision。路径必须是模型根目录内的相对路径。

Franka 工厂只接受模型根目录：

```python
from semantic_robot_sdk_franka import create_mujoco_sdk

sdk = create_mujoco_sdk(runtime_url, robot_id, "/opt/semantic-assets/franka-panda")
```

调用方不能再覆盖 URDF、Mesh root 或 `panda_hand`，避免 Runtime Profile 与 IK 使用
两套模型定义。

## 正式算法门控

```bash
FRANKA_MODEL_ROOT=/opt/semantic-assets/franka-panda make test-franka
```

该测试真实加载 URDF 和碰撞 Mesh，并使用 Pinocchio 计算非零 FK/IK，再用 Ruckig
生成受关节速度、加速度和加加速度限制的轨迹。未设置环境变量、模型缺失、许可证
资料不全或算法依赖不可用都会失败，不会用 `skip` 伪装成已经验收。
