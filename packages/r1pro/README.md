# semantic-robot-sdk-r1pro

R1 Pro 的稳定 modules、Fake/MuJoCo/real/Isaac Backend，以及本地/厂商 Provider 装配。

## BEHAVIOR 完整开发观测

使用 Semantic 场景部署给出的 endpoint、机器人 ID、场景 UUID 和 generation：

```python
from semantic_robot_sdk_r1pro.backends.isaac import IsaacBackend

backend = IsaacBackend(endpoint, robot_id)
observations = backend.development_observations(scene_instance_id, generation)
state = observations.state()
header, arrays = observations.capture_rgbd("head")
rgb, depth_m = arrays["rgb"], arrays["depth"]
calibration = header["calibration"]
geometry = observations.get_scene_geometry()
localization = observations.get_localization()
left_eef = observations.get_ee_pose("left", frame_id="body")
gripper = observations.get_gripper_state("left")
```

RGBD 及二进制策略观测需要 `isaac` extra（NumPy / SciPy）。标定随图像返回，也可通过
`get_camera_calibration("head", calibration_id)` 读取原帧。末端接口支持
`body`、`base_link`、`torso_link4`、`world`；输出 `data` 保持现用 Ability 的
PoseStamped 字段格式，夹爪保留米、米/秒和牛顿读回字段。

`capture_rgbd_frames()` 返回现有感知 Ability 所需的 RGB/深度 SensorFrame，包含
同帧 timestamp、step_count、内参和 USD 相机外参，可交给 `validate_pair()`。
该客户端只读 Runtime 数据；现用 Ability 的整体 SDK 门面及原子运动控制迁移仍需接入。
客户端不创建仿真、推流或 reset。
reset 后应使用新 generation 创建观测客户端。旧客户端会收到明确错误。
定位及碰撞网格属于完整开发观测，其中定位标记 `simulation_ground_truth`。


## BEHAVIOR 原子控制门面

Isaac 额外依赖安装后，可用 `from galaxea_r1pro import R1ProSDK, RobotProfile, BackendKind`
连接 Semantic 管理的场景。Profile.options 必须包含 `runtime_endpoint`、
`scene_instance_id`（UUID）和 `generation`，可选 `request_timeout`。
`R1ProSDK.open(profile)` 返回现有 Ability 使用的 joint / eef / torso / gripper /
woosh_move / state / sensors / command / safety 接口。

控制请求通过 HTTP 送到 Runtime；实测位置、速度和原生 episode 结果决定动作状态。
SDK 的 close 只断开客户端。仿真启动/停止/reset 和相机推流由 Semantic Runtime 管理。
重置后重新绑定新 generation。底盘反馈来源为 development_full 模式下的场景真值。
现有定时策略后端也可通过 `backend.atomic_controls(scene_instance_id, generation)`
取得同一门面。两种控制方式由 Runtime 互斥调度。

原子协议、字段及原生动作验证脚本见相邻 Isaac Runtime 的 `docs-atomic-sdk.md`。
