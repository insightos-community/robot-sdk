"""Host-owned planning jobs. Host invokes tick on its simulation thread."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time
import numpy as np
from .curobo_motion import CuroboPlanner, MotionPlanningError


class CuroboJobs:
    def __init__(self, runtime, planner_factory=CuroboPlanner, planning_timeout=600.0):
        self.runtime = runtime
        self.planner_factory = planner_factory
        self.planner = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sdk-curobo")
        self.future = None
        self.planning_timeout = planning_timeout
        self.prepared_plan = None

    @property
    def busy(self):
        return self.future is not None and not self.future.done()

    def close(self):
        self.prepared_plan = None
        self.pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _install(execution, result):
        execution.update(
            trajectory=result["positions"],
            trajectory_indices=np.asarray(result["indices"], dtype=int),
            trajectory_index=0,
            planning_wall_s=result.get("planning_wall_s", 0.0),
            trajectory_duration_s=(len(result["positions"]) - 1) * result["dt"],
            torso_used=result["torso_used"],
            pregrasp_planning=result.get("pregrasp_planning", {}),
            support_departure=result.get("support_departure", {}),
        )

    def tick(self, execution, joint_targets):
        rt = self.runtime
        future = execution.get("planning_future")
        if "trajectory" not in execution:
            if future is None:
                if self.busy:
                    raise MotionPlanningError("planner_busy", "上一次规划仍在退出")
                # Original command values identify the plan. Scene meshes, locked
                # joints and world-frame conversions are not compared or resampled.
                intent = dict(command=deepcopy(execution["planning_intent"]), dt=rt.dt)
                execution["plan_q"] = rt.q.copy()
                execution["planning_started"] = time.monotonic()
                cached, self.prepared_plan = self.prepared_plan, None
                if not execution["plan_only"] and cached and cached[0] == intent:
                    idx = np.asarray(cached[1]["indices"], dtype=int)
                    if np.max(np.abs(rt.q[idx] - cached[1]["positions"][0])) > 0.02:
                        execution["plan_reuse_reason"] = "start_mismatch"
                        raise MotionPlanningError("stale_plan", "当前关节偏离已规划轨迹起点")
                    self._install(execution, cached[1])
                    execution["plan_reused"] = True
                    execution["plan_reuse_reason"] = "prepared_plan"
                    execution["planning_wall_s"] = 0.0
                    execution["source_planning_wall_s"] = cached[1].get("planning_wall_s", 0.0)
                else:
                    execution["plan_reuse_reason"] = (
                        "plan_only"
                        if execution["plan_only"]
                        else "no_prepared_plan"
                        if cached is None
                        else "different_command"
                    )
                    if self.planner is None:
                        self.planner = self.planner_factory(rt)
                    request = self.planner.prepare(
                        execution["side"],
                        rt.eef_pose(execution["side"])
                        if "joint_goal" in execution
                        else execution["eef_goal"],
                        torso=execution["use_torso"],
                        approach_world=execution.get("approach_world"),
                        attached_object_ref=execution.get("attached_object_ref"),
                        contact_object_ref=execution.get("contact_object_ref"),
                        **(
                            {"joint_goal": execution["joint_goal"]}
                            if "joint_goal" in execution
                            else {}
                        ),
                        **(
                            {"allow_support_contact": True}
                            if execution.get("allow_support_contact")
                            else {}
                        ),
                    )
                    request["pregrasp_planner"] = execution.get("pregrasp_planner", "curobo")
                    execution["prepared_intent"] = intent
                    self.future = execution["planning_future"] = self.pool.submit(
                        self.planner.solve, request
                    )
                    return "planning"
            elif not future.done():
                if time.monotonic() - execution["planning_started"] > self.planning_timeout:
                    raise MotionPlanningError("planning_timeout", "cuRobo 规划超过墙钟预算")
                return "planning"
            else:
                result = future.result()
                idx = np.asarray(result["indices"], dtype=int)
                # Validate the actual start before feeding a collision-checked path.
                if np.max(np.abs(rt.q[idx] - execution["plan_q"][idx])) > 0.02:
                    raise MotionPlanningError("stale_plan", "规划期间活动关节偏离轨迹起点")
                self._install(execution, result)
                intent = execution.pop("prepared_intent")
                if execution["plan_only"]:
                    self.prepared_plan = (intent, result)
                    return "planned"
        cursor = execution["trajectory_index"]
        if cursor < len(execution["trajectory"]):
            idx = execution["trajectory_indices"]
            joint_targets[idx] = execution["trajectory"][cursor]
            execution["trajectory_index"] += 1
        return (
            "executing"
            if execution["trajectory_index"] < len(execution["trajectory"])
            else "settling"
        )

    @staticmethod
    def summary(execution):
        return dict(
            motion_mode="curobo",
            plan_only=execution["plan_only"],
            pregrasp_planning=execution.get("pregrasp_planning", {}),
            support_departure=execution.get("support_departure", {}),
            plan_reused=execution.get("plan_reused", False),
            plan_reuse_reason=execution.get("plan_reuse_reason", ""),
            source_planning_wall_s=execution.get("source_planning_wall_s", 0.0),
            torso_used=execution.get("torso_used", execution["use_torso"]),
            planning_wall_s=execution.get("planning_wall_s", 0.0),
            trajectory_points=len(execution.get("trajectory", [])),
            trajectory_index=execution.get("trajectory_index", 0),
            trajectory_duration_s=execution.get("trajectory_duration_s", 0.0),
        )
