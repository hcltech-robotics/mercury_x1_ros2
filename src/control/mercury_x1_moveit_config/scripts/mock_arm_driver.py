#!/usr/bin/env python3
"""Stand-in for the upper-body driver when testing MoveIt off the robot.

Serves {left_arm,right_arm,waist}_controller/follow_joint_trajectory and publishes /joint_states (50 Hz),
following each trajectory by its time stamps (linear between points). Starts in the "rest" pose.
"""
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_L = [f"joint{i}_L" for i in range(1, 8)]
ARM_R = [f"joint{i}_R" for i in range(1, 8)]
REST = dict(zip(ARM_L + ARM_R, [1.709, 1.419, -1.199, -0.581, -0.286, 0.264, 1.166,
                                -1.759, 1.641, 1.710, -0.539, -1.122, 0.559, 0.497]))
REST.update({"body": 0.0, "head": 0.0, "eye": 0.0})
CONTROLLERS = {"left_arm_controller": ARM_L, "right_arm_controller": ARM_R, "waist_controller": ["body"]}


class MockDriver(Node):
    def __init__(self) -> None:
        super().__init__("mercury_upper_body")
        self.pos = dict(REST)
        self.lock = threading.Lock()
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_timer(0.02, self.publish)
        cb = ReentrantCallbackGroup()
        self.servers = [ActionServer(self, FollowJointTrajectory, f"{c}/follow_joint_trajectory",
                                     execute_callback=self.execute, callback_group=cb) for c in CONTROLLERS]

    def publish(self) -> None:
        with self.lock:
            names, pos = list(self.pos), list(self.pos.values())
        js = JointState(name=names, position=pos)
        js.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(js)

    def execute(self, goal_handle):
        tr = goal_handle.request.trajectory
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in tr.points])
        pts = np.array([p.positions for p in tr.points])
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            q = np.array([np.interp(t, times, pts[:, j]) for j in range(len(tr.joint_names))])
            with self.lock:
                self.pos.update(zip(tr.joint_names, q.tolist()))
            if t >= times[-1]:
                break
            time.sleep(0.02)
        goal_handle.succeed()
        return FollowJointTrajectory.Result(error_code=FollowJointTrajectory.Result.SUCCESSFUL)


def main() -> None:
    rclpy.init()
    node = MockDriver()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    ex.spin()


if __name__ == "__main__":
    main()
