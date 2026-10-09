# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate held-object rigidity on the simulation owner thread."""


def require_rigid_attachment(obj):
    """Allow one link, or one connected assembly of enabled USD fixed joints.

    Inspect USD rather than obj.joints: native articulation views can omit fixed
    joints. No joint is locked or changed here. Movable joints are unsupported
    even if currently at rest or at a joint limit.
    """
    if len(obj.links) == 1:
        return  # Preserve the existing single-link path, including mesh fitting.

    from pxr import Usd, UsdPhysics
    from .curobo_motion import MotionPlanningError

    def reject(reason):
        raise MotionPlanningError(
            "unsupported_attached_object", f"持物 {obj.name} 不能作为固定整体附着: {reason}"
        )

    links = {str(link.prim_path) for link in obj.links.values()}
    if not links or len(links) != len(obj.links):
        reject("无有效 link 或 link 路径重复")
    graph = {path: set() for path in links}
    for prim in Usd.PrimRange(obj.prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        bodies = [list(rel.GetTargets()) for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]
        if any(len(body) != 1 for body in bodies):
            reject(f"{prim.GetPath()} 未连接两个内部 link")
        a, b = (str(body[0]) for body in bodies)
        if a not in links or b not in links or a == b:
            reject(f"{prim.GetPath()} 的连接不属于物体内部 link")
        if not prim.IsA(UsdPhysics.FixedJoint):
            reject(f"{prim.GetPath()} 是 {prim.GetTypeName()}，仅支持固定关节")
        graph[a].add(b)
        graph[b].add(a)

    pending = [next(iter(links))]
    connected = set()
    while pending:
        path = pending.pop()
        if path not in connected:
            connected.add(path)
            pending.extend(graph[path] - connected)
    if connected != links:
        reject("各 link 未由启用的固定关节连成一个整体")
