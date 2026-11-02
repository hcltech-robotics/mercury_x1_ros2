#!/usr/bin/env python3
"""Checks a running move_group (demo.launch.py off-robot, or the robot stack): self-collision, IK, planning.

  ros2 run mercury_x1_moveit_config check_moveit.py [--execute]

Plans only unless --execute (then it also runs the plans: on the robot the arms MOVE).
"""
import math
import sys

import rclpy
from builtin_interfaces.msg import Duration
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_L = [f"joint{i}_L" for i in range(1, 8)]
ARM_R = [f"joint{i}_R" for i in range(1, 8)]
REST_L = [1.709, 1.419, -1.199, -0.581, -0.286, 0.264, 1.166]
REST_R = [-1.759, 1.641, 1.710, -0.539, -1.122, 0.559, 0.497]
# left wrist (link5_L/link6_L) inside the right shoulder (link1_R) with the right arm at rest (found by sampling)
INTO_R = [0.816, 1.953, -0.677, -1.518, -1.215, -1.869, -0.39]


class Check(Node):
    def __init__(self) -> None:
        super().__init__("check_moveit")
        self.valid = self.create_client(GetStateValidity, "/check_state_validity")
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.fk = self.create_client(GetPositionFK, "/compute_fk")
        self.move = ActionClient(self, MoveGroup, "/move_action")
        for c in (self.valid, self.ik, self.fk):
            c.wait_for_service(timeout_sec=30)
        self.move.wait_for_server(timeout_sec=30)

    def call(self, client, req):
        f = client.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=30)
        return f.result()

    @staticmethod
    def state(names, values) -> RobotState:
        return RobotState(joint_state=JointState(name=names, position=values), is_diff=True)

    def check_state(self, label, names, values, group="both_arms"):
        r = self.call(self.valid, GetStateValidity.Request(robot_state=self.state(names, values), group_name=group))
        pairs = sorted({f"{c.contact_body_1}<->{c.contact_body_2}" for c in r.contacts})
        print(f"  {label:42s} {'valid' if r.valid else 'IN COLLISION'} {', '.join(pairs[:6])}")
        return r.valid

    def plan(self, label, group, names, values, execute=False):
        req = MotionPlanRequest(group_name=group, num_planning_attempts=4, allowed_planning_time=5.0,
                                max_velocity_scaling_factor=0.3, max_acceleration_scaling_factor=0.3)
        req.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=n, position=v, tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            for n, v in zip(names, values)])]
        goal = MoveGroup.Goal(request=req)
        goal.planning_options.plan_only = not execute
        f = self.move.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=30)
        rf = f.result().get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=120)
        res = rf.result().result
        tr = res.planned_trajectory.joint_trajectory
        dur = tr.points[-1].time_from_start.sec + tr.points[-1].time_from_start.nanosec * 1e-9 if tr.points else 0
        print(f"  {label:42s} code {res.error_code.val} ({'ok' if res.error_code.val == 1 else 'FAILED'}), "
              f"{len(tr.points)} points, {dur:.1f} s, planned in {res.planning_time:.2f} s")
        return res.error_code.val == 1


def main() -> None:
    execute = "--execute" in sys.argv
    rclpy.init()
    n = Check()
    print("self-collision:")
    n.check_state("rest (arms hanging)", ARM_L + ARM_R, REST_L + REST_R)
    n.check_state("all joints zero", ARM_L + ARM_R, [0.0] * 14)
    n.check_state("left wrist in the right shoulder", ARM_L + ARM_R, INTO_R + REST_R)
    print("IK (pose from FK of a reach-forward configuration):")
    reach_l = [0.0, 0.6, 0.0, -1.2, 0.0, 0.6, 0.0]
    fk = n.call(n.fk, GetPositionFK.Request(fk_link_names=["link7_L"],
                                            robot_state=n.state(ARM_L + ARM_R, reach_l + REST_R)))
    pose = fk.pose_stamped[0]
    p = pose.pose.position
    print(f"  link7_L target in {pose.header.frame_id}: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
    ik = n.call(n.ik, GetPositionIK.Request(ik_request=PositionIKRequest(
        group_name="left_arm", pose_stamped=pose, avoid_collisions=True,
        robot_state=n.state(ARM_L + ARM_R, REST_L + REST_R), timeout=Duration(sec=1))))
    sol = dict(zip(ik.solution.joint_state.name, ik.solution.joint_state.position))
    print(f"  IK code {ik.error_code.val} ({'ok' if ik.error_code.val == 1 else 'FAILED'})",
          [round(math.degrees(sol[j]), 1) for j in ARM_L] if ik.error_code.val == 1 else "")
    print("planning:")
    n.plan("left arm: rest -> reach forward", "left_arm", ARM_L, reach_l, execute)
    n.plan("both arms: -> rest", "both_arms", ARM_L + ARM_R, REST_L + REST_R, execute)
    n.plan("left wrist into right shoulder (must fail)", "left_arm", ARM_L, INTO_R, False)
    n.plan("waist: -> 30 deg left", "waist", ["body"], [math.radians(30)], execute)
    if execute:
        n.plan("waist: -> straight", "waist", ["body"], [0.0], execute)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
