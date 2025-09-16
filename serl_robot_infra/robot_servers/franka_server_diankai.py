"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
from flask import Flask, request, jsonify
import numpy as np

# import rospy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from genie_msgs.msg import EndState
import threading
from rclpy.executors import MultiThreadedExecutor


import time
import subprocess
from scipy.spatial.transform import Rotation as R
from absl import app, flags

# from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
# from franka_msgs.srv import SetLoad
# from serl_franka_controllers.msg import ZeroJacobian
import geometry_msgs.msg as geom_msg
# from dynamic_reconfigure.client import Client as ReconfClient


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "10.42.0.101", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "10.42.0.101", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "None", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string("flask_url", 
    "127.0.0.1",
    "URL for the flask server to run on."
)
flags.DEFINE_string("ros_port", "11311", "Port for the ROS master to run on.")


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, '/wbc/arm_command', 100)
        self.joint_states = JointState()
        self.initialize_joint_state()

    def publish_message(self,joint_positions):
        self.joint_states.header.stamp = self.get_clock().now().to_msg()
        self.joint_states.position=joint_positions
        self.publisher_.publish(self.joint_states)
        #self.get_logger().info(f'关节发布的消息: {self.joint_states.position}')

    def initialize_joint_state(self):
        self.joint_states.name = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5',
            'joint6', 'joint7', 'joint8', 'joint9', 'joint10',
            'joint11', 'joint12', 'joint13', 'joint14'
        ]
        self.joint_states.velocity = []
        self.joint_states.effort = []


class GripperCommandPublisher(Node):
    def __init__(self):
        super().__init__('gripper_command_publisher')
        self.publisher_ = self.create_publisher(JointState, '/wbc/right_ee_command', 100)
        self.gripper_states = JointState()
        self.initialize_gripper_state()

    def publish_message(self,gripper_positions):
        self.gripper_states.header.stamp = self.get_clock().now().to_msg()
        self.gripper_states.position = gripper_positions
        self.publisher_.publish(self.gripper_states)
        #self.get_logger().info(f'夹爪发布的消息: {self.gripper_states.position}')

    def initialize_gripper_state(self):
        self.gripper_states.name = ['right']
        self.gripper_states.velocity = []
        self.gripper_states.effort = []
        
        
class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        self.position = None  # 初始化 position
        self.time = None
        self.lock = threading.Lock()  # 用于保护对 joint_state 的访问
        self.subscription = self.create_subscription(
            JointState,
            '/hal/arm_joint_state',
            self.joint_state_callback,
            10
        )
        self.subscription 

    def joint_state_callback(self, msg):
        with self.lock:
            #print("msg",msg.position)
            self.position = msg.position  # 每次更新 position
            self.time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def get_position(self):
        with self.lock:
            return self.position
        
    def get_time(self):
        with self.lock:
            return self.time  # 返回时间戳
        
class GripperStateSubscriber(Node):
    def __init__(self):
        super().__init__('gripper_state_subscriber')
        self.position = None  # 初始化 position
        self.time = None
        self.lock = threading.Lock()  # 用于保护对 gripper_state 的访问
        self.subscription = self.create_subscription(
            EndState,
            '/hal/right_ee_data',
            self.gripper_state_callback,
            10
        )
        self.subscription 

    def gripper_state_callback(self, msg):
        with self.lock:
            #print("*******")
            #print("msg",msg.end_state[0].position)
            self.position = [0,msg.end_state[0].position]  # 每次更新 position
            self.time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def get_position(self):
        with self.lock:
            return self.position  # 返回 position
        
    def get_time(self):
        with self.lock:
            return self.time  # 返回时间戳

# Interpolate the actions to make the robot move smoothly
def interpolate_action_dualarm(prev_action, cur_action):
    #steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    steps=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 20,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 20])
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

    


# Main loop for the manipulation task
def move_only(joint_node, joint_pub_node, gripper_node, gripper_pub_node, right_joint, right_gripper):
    try:
        #发布频率
        #target_frequency = args.publish_rate
        target_frequency = 30
        period = 1 / target_frequency  # 计算周期时间（秒）
        time.sleep(1)
        
        # Initialize the previous action to be the initial robot state
        left_joint = [-1.7635668516159058, 0.752356767654419, -1.0365300178527832, 0.9499779939651489, -2.9094560146331787, 0.5424855947494507, -2.0491011142730713]
        left_gripper =[0.0]
        
        #获得当前关节位置并插值   运行到初始化位置  
        joint_position = joint_node.get_position()  # 获取位置
        gripper_position = gripper_node.get_position()  
        joint_position.insert(7, gripper_position[0])
        joint_position.insert(15, gripper_position[1])
        pre_action = joint_position
        # print("pre_action",pre_action)
        action=np.array(left_joint+left_gripper+right_joint+right_gripper)
        # print("action",action)
        
        # exit(0)
        
        assert len(pre_action)==len(action)==16,"len(pre_action)==len(action)==16"
        
        interp_actions = interpolate_action_dualarm(pre_action, action)
        
        

        for act in interp_actions:
            start_time = time.perf_counter()
            #gripper_positions = [float(act[7]),float(act[15])]  # 确保是浮动类型
            gripper_positions = [float(act[15])] 
            #print("gripper_positions",gripper_positions)
            joint_positions = [float(act[i]) for i in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]] 
            #print("joint_positions",joint_positions)
            gripper_positions = [x / 120.0 for x in gripper_positions]  # 将每个值从 [0, 120] 缩放到 [0, 1]

            joint_pub_node.publish_message(joint_positions)
            gripper_pub_node.publish_message(gripper_positions)
            
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time < period:
                time.sleep(period - elapsed_time)


        print("完成")
    except KeyboardInterrupt:
        print("程序被用户中断")

            


def main(_):
    try: 
        ROS_PKG_NAME = "serl_franka_controllers"

        ROBOT_IP = FLAGS.robot_ip
        GRIPPER_IP = FLAGS.gripper_ip
        GRIPPER_TYPE = FLAGS.gripper_type
        RESET_JOINT_TARGET = FLAGS.reset_joint_target

        webapp = Flask(__name__)


        rclpy.init()
        joint_node = JointStateSubscriber()
        gripper_node=GripperStateSubscriber()
        # 创建 MultiThreadedExecutor 来管理多个节点
        def spin_executor(executor):
            executor.spin()
        executor = MultiThreadedExecutor()
        executor.add_node(joint_node)
        executor.add_node(gripper_node)
        # 在后台线程中运行 ROS 事件循环
        spin_thread = threading.Thread(target=spin_executor, args=(executor,), daemon=True)
        spin_thread.start()
        
        joint_pub_node = JointStatePublisher()
        gripper_pub_node = GripperCommandPublisher()
        
        
        @webapp.route("/getposition", methods=["POST"])
        def get_position():
            return jsonify({"position": np.array(joint_node.position).tolist()})
        
        # Route for Sending a pose command
        @webapp.route("/pose", methods=["POST"])
        def pose():
            pos = np.array(request.json["arr"])
            # print("Moving to", pos)
            joint_pub_node.publish_message(pos)
            return "Moved"

        # Route for getting gripper distance
        @webapp.route("/get_gripper", methods=["POST"])
        def get_gripper():
            return jsonify({"gripper": gripper_node.position})


        # Route for moving the gripper
        @webapp.route("/move_gripper", methods=["POST"])
        def move_gripper():
            gripper_pos = request.json
            print(gripper_pos)
            # pos = np.clip(float(gripper_pos["gripper_pos"]), 0, 120)  # 0-255
            print(f"move gripper to {gripper_pos}")
            gripper_pos = [float(x) / 120.0 for x in gripper_pos]
            gripper_pub_node.publish_message(gripper_pos)
            return "Moved Gripper"
        
        # Route for moving the gripper
        @webapp.route("/move_arm", methods=["POST"])
        def move_arm():
            arm_pos = request.json
            right_joint = arm_pos[0]
            right_gripper = arm_pos[1]
            print(right_joint)
            print(right_gripper)
            # pos = np.clip(float(gripper_pos["gripper_pos"]), 0, 120)  # 0-255
            move_only(joint_node, joint_pub_node, gripper_node, gripper_pub_node, right_joint, right_gripper)
            return "Moved Gripper"




        # # Route for Running Joint Reset
        # @webapp.route("/jointreset", methods=["POST"])
        # def joint_reset():
        #     robot_server.clear()
        #     robot_server.reset_joint()
        #     return "Reset Joint"

        # # Route for Resetting the Gripper. It will reset and activate the gripper
        # @webapp.route("/reset_gripper", methods=["POST"])
        # def reset_gripper():
        #     print("reset gripper")
        #     gripper_server.reset_gripper()
        #     return "Reset"

        # # Route for Opening the Gripper
        # @webapp.route("/open_gripper", methods=["POST"])
        # def open():
        #     print("open")
        #     gripper_server.open()
        #     return "Opened"

        # # Route for getting all state information
        # @webapp.route("/getstate", methods=["POST"])
        # def get_state():
        #     return jsonify(
        #         {
        #             "pose": np.array(robot_server.pos).tolist(),
        #             "vel": np.array(robot_server.vel).tolist(),
        #             "force": np.array(robot_server.force).tolist(),
        #             "torque": np.array(robot_server.torque).tolist(),
        #             "q": np.array(robot_server.q).tolist(),
        #             "dq": np.array(robot_server.dq).tolist(),
        #             "jacobian": np.array(robot_server.jacobian).tolist(),
        #             "gripper_pos": gripper_server.gripper_pos,
        #         }
        #     )

        webapp.run(host=FLAGS.flask_url)
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 确保在退出时关闭 ROS 节点
        print("关闭节点...")
        joint_pub_node.destroy_node()
        gripper_pub_node.destroy_node()
        # 1. 关闭执行器
        executor.shutdown()
        # 2. 移除节点
        executor.remove_node(joint_node)
        executor.remove_node(gripper_node)        
        # 3. 销毁节点
        joint_node.destroy_node()
        gripper_node.destroy_node()
        
        # 4. 等待线程结束
        spin_thread.join(timeout=5.0)
            
        # 5. 关闭 ROS 上下文
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    app.run(main)
