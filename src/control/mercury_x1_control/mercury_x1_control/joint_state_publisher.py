import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
from pymycobot.mercury_arms_socket import MercuryArmsSocket
import time

class MercuryJointStatePublisher(Node):
    def __init__(self):
        super().__init__('mercury_joint_state_publisher')
        
        # Initialize Mercury arms
        self.left_arm = MercuryArmsSocket('left_arm', ip="127.0.0.1")
        self.right_arm = MercuryArmsSocket('right_arm', ip="127.0.0.1")
        
        # Power on arms
        self.get_logger().info("Powering on left arm...\r\n") 
        if self.left_arm.is_power_on() != 1:
            self.left_arm.power_on()
        if self.right_arm.is_power_on() != 1:
            self.right_arm.power_on() 
        
        # Create publisher
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        
        # Timer to publish at 20 Hz
        self.timer = self.create_timer(0.05, self.publish_joint_states)
        
        self.get_logger().info('Mercury joint state publisher started')

    def publish_joint_states(self):
        try:
            # Get current angles from both arms (returns degrees)
            left_angles = self.left_arm.get_angles()
            right_angles = self.right_arm.get_angles()
            # Get individual angles
            jaw_angle = self.right_arm.get_angle(11)
            neck_angle = self.right_arm.get_angle(12) 
            waist_angle = self.right_arm.get_angle(13)

            # Check if ANY value is None
            if (left_angles is None or 
                right_angles is None):
                # self.get_logger().warn(f"Failed to read joint angles: {left_angles is None} {right_angles is None} {jaw_angle is None} {neck_angle is None} {waist_angle is None}")
                return

            # Now create the list with valid values
            jaw_neck_waist_angles = [jaw_angle, neck_angle, waist_angle]
                
            # Convert degrees to radians (ROS standard)
            left_rad = [math.radians(angle) for angle in left_angles]
            right_rad = [math.radians(angle) for angle in right_angles]
            jaw_neck_waist_rad =  [math.radians(angle) for angle in jaw_neck_waist_angles]
            
            # Create JointState message
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            
            # Joint names must match your URDF exactly!
            msg.name = [
                'joint1_L', 'joint2_L', 'joint3_L', 'joint4_L', 'joint5_L', 'joint6_L', 'joint7_L',
                'joint1_R', 'joint2_R', 'joint3_R', 'joint4_R', 'joint5_R', 'joint6_R', 'joint7_R'
            ]
            
            # Combine left and right arm angles
            msg.position = left_rad + right_rad
            msg.velocity = []  # Optional: leave empty if not available
            msg.effort = []    # Optional: leave empty if not available
            
            # Publish
            self.publisher_.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing joint states: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    publisher = MercuryJointStatePublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        publisher.destroy_node()

if __name__ == '__main__':
    main()