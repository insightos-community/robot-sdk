# semantic-robot-sdk-core

[English](README.md) | [简体中文](README.zh-CN.md)

公共 Robot SDK 类型、Backend/Provider 接口、统一部署配置和 Fake 测试基础。不包含具体机器人型号或规划算法。

## 可选定时控制

`semantic_robot_sdk_core.control` 提供 `ControlSequence` 和 `TimedControlBackend`。
该接口独立于现有 `RobotBackend.execute_plan`，不要求旧型号模拟策略控制。

控制序列按组表达绝对关节角、末端位移/轴角增量、底盘速度和夹爪开合方向。
每段携带 Robot、执行身份、generation 和控制周期；模型原始数组必须在型号
适配层转换后才能提交。`cancel_execution` 撤销整个执行身份，包含尚未返回的
推理结果，而不仅是某一个 chunk 的命令。序列完成只表示控制已消费，任务验收
仍由 Robot Skill 完成。

Franka MuJoCo Backend 新增该接口和 `synchronized_observation`，后者保留原始
RGB、EEF 及两指关节状态，不经过 Web 图像处理。实际能力以 Runtime Profile
声明为准；LIBERO 的 OSC 配置与原有 JOINT_POSITION 配置分开部署。
