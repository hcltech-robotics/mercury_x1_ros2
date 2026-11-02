"""Arm poses from the control page through MoveIt: plan with self-collision checks, then execute.

The named poses are the SRDF group_states of mercury_x1_moveit_config (rest, reach_forward_L, ...); a pose of
group "both_arms" moves both arms in one plan. Goals go to move_group's /move_action (plan + execute); its
trajectories run on the upper-body driver's FollowJointTrajectory servers. Stop cancels the goal (MoveIt cancels
the execution, the driver stops the arm) and also calls /upper_body/stop.
All methods run on the ROS thread (WebNode.post).
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest
from rclpy.action import ActionClient

# moveit_msgs/MoveItErrorCodes -> words for the page
ERRORS = {1: "done", 99999: "failed", -1: "no path found", -2: "invalid plan", -4: "arm did not follow (driver)",
          -6: "timed out", -7: "preempted", -10: "start pose is in collision", -12: "goal pose is in collision",
          -13: "goal not reached", -16: "invalid goal", -31: "no IK solution", -2147483648: "?"}
GROUP_LABEL = {"both_arms": "both arms", "left_arm": "left arm", "right_arm": "right arm", "waist": "waist",
               "left_arm_waist": "left arm + waist", "right_arm_waist": "right arm + waist"}


def load_poses() -> list[dict]:
    try:
        srdf = os.path.join(get_package_share_directory("mercury_x1_moveit_config"), "config", "mercury_x1.srdf")
    except PackageNotFoundError:
        return []
    out = []
    for gs in ET.parse(srdf).getroot().findall("group_state"):
        joints = {j.get("name"): float(j.get("value")) for j in gs.findall("joint")}
        name, group = gs.get("name"), gs.get("group")
        label = name.replace("_L", "").replace("_R", "").replace("_", " ")
        out.append({"name": name, "group": group, "label": f"{label} · {GROUP_LABEL.get(group, group)}", "joints": joints})
    return out


class ArmMotion:
    def __init__(self, node) -> None:
        self.node = node
        self.client = ActionClient(node, MoveGroup, "/move_action")
        self.poses = load_poses()
        self.goal_handle = None
        self.state = "idle"          # idle | planning | moving | done | failed | stopped
        self.message = ""
        self.pose = ""
        self.t = 0.0

    def snapshot(self) -> dict:
        return {"state": self.state, "message": self.message, "pose": self.pose, "t": self.t,
                "ready": self.client.server_is_ready(),
                "poses": [{k: p[k] for k in ("name", "group", "label")} for p in self.poses]}

    def _set(self, state: str, message: str = "") -> None:
        self.state, self.message, self.t = state, message, time.time()

    def goto(self, name: str, speed: float) -> None:
        pose = next((p for p in self.poses if p["name"] == name), None)
        if pose is None:
            self._set("failed", f"unknown pose {name}")
            return
        if not self.client.server_is_ready():
            self._set("failed", "MoveIt is not running (switch on Motion planning)")
            return
        if self.state in ("planning", "moving"):
            self._set("failed", "still moving - press Stop first")
            return
        speed = max(0.05, min(0.5, speed))
        req = MotionPlanRequest(group_name=pose["group"], num_planning_attempts=3, allowed_planning_time=3.0,
                                max_velocity_scaling_factor=speed, max_acceleration_scaling_factor=speed)
        req.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=j, position=v, tolerance_above=0.005, tolerance_below=0.005, weight=1.0)
            for j, v in pose["joints"].items()])]
        goal = MoveGroup.Goal(request=req)
        goal.planning_options.plan_only = False
        self.pose = pose["label"]
        self._set("planning", f"{pose['label']} at {round(speed * 100)} %")
        self.client.send_goal_async(goal, feedback_callback=self._feedback).add_done_callback(self._accepted)

    def _feedback(self, fb) -> None:
        st = fb.feedback.state          # "PLANNING", "MONITOR" (executing), "IDLE"
        if st == "MONITOR" and self.state == "planning":
            self._set("moving", self.message)

    def _accepted(self, fut) -> None:
        gh = fut.result()
        if gh is None or not gh.accepted:
            self._set("failed", "MoveIt refused the goal")
            return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(self._done)

    def _done(self, fut) -> None:
        self.goal_handle = None
        if self.state == "stopped":
            return
        code = fut.result().result.error_code.val
        self._set("done" if code == 1 else "failed", ERRORS.get(code, f"error {code}"))

    def stop(self) -> None:
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        if self.state in ("planning", "moving"):
            self._set("stopped", "stopped")
