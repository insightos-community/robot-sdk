# Semantic Robot SDK

本仓库维护 Robot SDK 的公共规范和各机器人型号实现。根目录只是开发工作区，不发布
包含所有型号的混合 Wheel：

```text
packages/core    semantic-robot-sdk-core
packages/r1pro   semantic-robot-sdk-r1pro
packages/franka  semantic-robot-sdk-franka
```

`core` 只提供公共类型、错误、Backend/Provider 接口、命令反馈与停止、统一部署配置、
订阅协议和 Fake 共同测试。机器人型号、关节名称、URDF、Pinocchio、Ruckig、导航算法、
厂商接口和 MuJoCo HTTP 路径均留在对应型号包。

## 安装和引用

Ability 环境安装公共 Core 与当前 Robot 型号对应的 Wheel，并通过正常 Python Import
使用 SDK；正式部署不把本仓库源码目录加入 `PYTHONPATH`。

```bash
python -m pip install \
  dist/semantic_robot_sdk_core-*.whl \
  dist/semantic_robot_sdk_r1pro-*.whl
```

```python
from semantic_robot_sdk_r1pro import R1ProSDK

with R1ProSDK.from_environment() as robot:
    state = robot.state.snapshot()
    frame = robot.sensors.latest("camera.rgb")
```

同型号机器人共享同一组 Wheel；每台 Robot 通过独立的 `RobotDeployment` 和运行数据目录
隔离身份、连接、命令和状态。

## R1 Pro 结构

原生 cuRobo 的单 link 与固定多 link 持物支持见
[固定多 link 附着说明](packages/r1pro/FIXED_ATTACHMENTS.md)。

```text
semantic_robot_sdk_r1pro
├── modules       Ability 使用的稳定资源接口
├── backends      fake / mujoco / real / isaac
├── providers     local/vendor kinematics、motion、navigation
├── profile.py    R1 Pro 固定型号能力
└── sdk.py        按部署配置装配以上组件
```

- Backend 只访问状态、底层轨迹、夹爪、传感器、stop 和 hold。
- Provider 负责 IK、轨迹生成和路径规划；MuJoCo Runtime 不承担这些算法。
- `real` 必须注入真实厂商驱动；`isaac` 在 v0.5 明确返回未实现。
- Fake 提供命令去重、反馈顺序、资源互斥、状态变化和停止证据。

Fake 抓取环境通过 RobotDeployment 的 `robot.sdk.options.initial_grasp_targets` 声明。
共享状态文件第一次创建时，Robot SDK 会自动写入这些环境事实；七个 Ability 后续
启动或重启只读取同一状态，不会重复初始化。底层测试仍可直接调用
`configure_grasp_fixture(...)` 或 `set_tool_contact_state(...)` 设置接触测试状态。此后
关闭和释放只更新工具接触、力与 `contact` SensorFrame；稳定承载和抓取成功仍由
Ability根据连续状态判断。没有fixture时关闭夹爪不会伪造接触成功。

## 一台 Robot 一份配置

Ability 和 Pilot 统一读取：

```bash
export SEMANTIC_ROBOT_CONFIG=/etc/semantic/robots/r1pro-001/robot-deployment.yaml
```

配置样例位于 `examples/robot-deployment.fake.yaml` 和
`examples/robot-deployment.mujoco.yaml`。Endpoint、固件和 Provider 只写在这份部署
配置中，不重复写入每个 Ability CR。

同一主机运行多个仿真 Robot 时，每组 Pilot、AbilityFramework 和 Ability 可以覆盖
自己的 Runtime 地址。环境变量优先于 YAML 中的 `robot.sdk.endpoint`：

```bash
SEMANTIC_ROBOT_CONFIG=/etc/semantic/robots/r1pro-sim.yaml \
SEMANTIC_ROBOT_SDK_ENDPOINT=http://127.0.0.1:8091 \
semantic-pilot ...
```

如果多个虚拟 Robot 位于同一个 Runtime，它们可以使用相同 Endpoint，并由各自的
`robot.id` 精确路由：

```yaml
# r1pro-001/robot-deployment.yaml
robot:
  id: r1pro-001
  model: r1_pro_chassis
  backend: mujoco
  sdk:
    package: semantic-robot-sdk-r1pro
    endpoint: http://127.0.0.1:8090
```

```yaml
# r1pro-002/robot-deployment.yaml
robot:
  id: r1pro-002
  model: r1_pro_chassis
  backend: mujoco
  sdk:
    package: semantic-robot-sdk-r1pro
    endpoint: http://127.0.0.1:8090
```

如果 Robot 位于不同 Runtime，每组进程使用不同的 `SEMANTIC_ROBOT_SDK_ENDPOINT`
即可覆盖模板地址。Robot ID、坐标系、Provider 和安全限制仍由部署配置保存；环境变量
只覆盖 Endpoint，不改变 Robot 身份。

同一型号的多个 Robot 实例不会复制 Wheel。正式部署由机器人类型包共享 SDK 制品，
只在每台 Robot 的实例目录生成 `robot-deployment.yaml`。

Ability 使用的 v0.5 稳定入口如下：

- `state.capabilities()`、`state.snapshot()`。
- `base.plan_route(goal, occupancy=None, minimum_clearance_m=None)`、
  `base.follow_route(plan, command_id=...)`、
  `base.navigate(goal, command_id=..., occupancy=None, minimum_clearance_m=None)`。
- `upper_body.plan_joints(...)`、`move_joints(...)`、`plan_end_effector(...)` 和
  `move_end_effector(...)`。
- `end_effector.set_opening(...)`、`close_until_contact(...)`、`release(...)`。
- `sensors.list()`、`latest(...)`、`subscribe(...)` 和 `subscribe_state(...)`。
- `commands.get(...)`、`feedback(...)`、`stop(...)`。
- `safety.stop_and_hold(...)`、`hold()`。

本地导航的 `occupancy` 是可选覆盖参数，不要求 Navigation Ability 构造 SDK 内部栅格。
SDK 优先使用装配时注入的 `NavigationMapSource`，也可读取
`robot.sdk.options.navigation_map_source` 中的固定场景地图。Fake 后端在没有配置时使用
明确标记的测试空地图；MuJoCo 和真机使用本地规划却没有地图来源时会明确失败。厂商
Navigation Provider 可以从厂商导航服务自行取图。

运动接口始终使用调用方提供的稳定 `command_id`。重复 ID 与相同内容返回原命令；
内容不同会被拒绝。停止通过 `robot.safety.stop_and_hold(command_id)` 进入底层并验证
hold；无法确认物理状态时返回 `interrupted`。

## MuJoCo 边界

R1 Pro MuJoCo Backend 只消费 `plugin-mujoco-v040-platform` 已有的 Robot Profile、
state、command、stop/hold、sensor 和 WebSocket 流接口。本仓库不修改或内嵌 Runtime。
真实组合测试由调用方先启动固定版本 Runtime：

```bash
PLUGIN_MUJOCO_URL=http://127.0.0.1:8090 \
R1PRO_ASSET_ROOT=/path/to/mujoco_asset make test-plugin
```

## 开发与构建

```bash
uv sync --all-packages --group test
make lint
make test
make build
```

`make build` 分别生成三个 Wheel。模型与 Runtime 专项测试不会因缺少外部资产而伪装
通过；运行 `make test-r1pro`、`make test-franka` 或 `make test-plugin` 前必须显式提供
对应环境变量。
