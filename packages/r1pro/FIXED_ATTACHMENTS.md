# cuRobo 固定多 link 持物

原生 Runtime 使用 `CuroboPlanner.prepare()` 在仿真主线程生成数值快照。
持物身份仍使用原生辅助抓持结果，Skill 输入和 Runtime API 不变。

单 link 物体沿用原来的附着流程。多个 link 时，`curobo_attachments.py`
读取物体 USD 子树中的关节，要求所有 link 通过启用的 `PhysicsFixedJoint`
连成一个整体。可动关节、未连接的 link、连接物体外部或世界的关节继续返回
`unsupported_attached_object`；即使可动关节当前静止也不会作为固定连接处理。
不使用 `obj.joints` 判断固定连接，因为原生 articulation 视图可能省略固定关节。

`scene_geometry()` 已包含所有 link 启用的碰撞网格及其世界坐标顶点。
`prepare()` 将它们转换到同一个规划基坐标系，并保留每个网格名称。
数值线程沿用 cuRobo 的 `attach_objects_to_robot(..., merge_meshes=True)`，
根据实测 EEF 位姿将全部网格合并为夹爪局部附着球，同时禁用对应世界障碍。
规划清理时 detach 恢复全部对应障碍；释放后的新快照不再附着该物体，并读取
其当前位置。无碰撞几何的固定元数据 link 不需要虚构网格。

已检查的收音机 `wxnicr` 使用两个 link：主体和按钮元数据 link，二者之间
为固定关节。资产现有 14 个碰撞网格均属于主体，按钮 link 没有碰撞网格。

`tests/conformance/test_curobo_attachments.py` 覆盖单 link 兼容、固定连接链、
可动/禁用/断开/外部连接拒绝、旋转和平移坐标变换、全量网格附着及释放后快照。
USD 相关测试需要可导入的 `pxr.Usd` / `pxr.UsdPhysics`；缺失时会跳过这些测试。

本机原生部署通过 `PYTHONPATH` 加载 SDK 源码，修改后需要重启 Runtime 生效。
仅完成几何和附着验证不等于完整携物轨迹规划或真实执行成功。
