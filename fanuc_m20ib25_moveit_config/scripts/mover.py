#!/usr/bin/env python

from __future__ import print_function

import rospy

import sys
import copy
import numpy as np
import math
import moveit_commander

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from fanuc_m20ib25_msgs.srv import MoverService_PickNPlace, MoverService_UsingPlanningSequencer, MoverService_PickNPlaceRequest, MoverService_PickNPlaceResponse, MoverService_UsingPlanningSequencerRequest, MoverService_UsingPlanningSequencerResponse
from fanuc_m20ib25_msgs.msg import FanucM20iB25UnityTargetPoseToROS, FanucM20iB25ROSJointsToUnity, FanucM20iB25UnityRobotJoints

import tf.transformations as transformations

joint_names = ['Revolute1', 'Revolute2', 'Revolute3', 'Revolute4', 'Revolute5', 'Revolute6']

# Between Melodic and Noetic, the return type of plan() changed. moveit_commander has no __version__ variable, so checking the python version as a proxy
if sys.version_info >= (3, 0):
    def planCompat(plan):
        return plan[1]
else:
    def planCompat(plan):
        return plan
        

def rotate_pose(pose, roll, pitch, yaw):
    # Convert these angle to radians
    roll = np.deg2rad([roll])
    pitch = np.deg2rad([pitch])
    yaw = np.deg2rad([yaw])
    # Create a rotation matrix from roll, pitch, and yaw angles
    rotation_matrix = transformations.euler_matrix(roll, pitch, yaw)

    # Extract the current pose orientation as a quaternion
    current_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

    # Create the current pose transformation matrix
    current_pose_matrix = transformations.quaternion_matrix(current_orientation)
    current_pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

    # Apply the rotation to the current pose
    new_pose_matrix = transformations.concatenate_matrices(rotation_matrix, current_pose_matrix)

    # Extract the new position and orientation
    new_position = new_pose_matrix[:3, 3]
    new_orientation = transformations.quaternion_from_matrix(new_pose_matrix)

    # Create a new Pose object for the transformed pose
    new_pose = Pose()
    new_pose.position.x = new_position[0]
    new_pose.position.y = new_position[1]
    new_pose.position.z = new_position[2]
    new_pose.orientation.x = new_orientation[0]
    new_pose.orientation.y = new_orientation[1]
    new_pose.orientation.z = new_orientation[2]
    new_pose.orientation.w = new_orientation[3]

    return new_pose

    
"""
    Given the start angles of the robot, plan a trajectory that ends at the destination pose.
    * To account for the rotation of the robot arm's base, the base rotation angle into the destination pose. 
    This involves rotating the destination pose around the base of the robot by the specified angle before
    planning the trajectory. You can achieve this by transforming the destination pose using a rotation matrix
    as done below using function 'rotate_pose'
"""
def plan_trajectory(move_group, destination_pose, start_joint_angles, robot_arm_rotation): 
    current_joint_state = JointState()
    current_joint_state.name = joint_names
    current_joint_state.position = start_joint_angles

    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = current_joint_state
    move_group.set_start_state(moveit_robot_state)

    # move_group.set_pose_target(destination_pose)
    rotated_destination_pose = rotate_pose(destination_pose, robot_arm_rotation.x, robot_arm_rotation.y, robot_arm_rotation.z)
    move_group.set_pose_target(rotated_destination_pose)

    plan = move_group.plan()

    if not plan:
        exception_str = """
            Trajectory could not be planned for a destination of {} with starting joint angles {}.
            Please make sure target and destination are reachable by the robot.
        """.format(destination_pose, destination_pose)
        raise Exception(exception_str)
    
    if plan:
        move_group.go()

    return planCompat(plan)


"""
    Creates a pick and place plan using the four states below.
    
    1. Pre Grasp - position gripper directly above target object
    2. Grasp - lower gripper so that fingers are on either side of object
    3. Pick Up - raise gripper back to the pre grasp position
    4. Place - move gripper to desired placement position

    Gripper behaviour is handled outside of this trajectory planning.
        - Gripper close occurs after 'grasp' position has been achieved
        - Gripper open occurs after 'place' position has been achieved

    https://github.com/ros-planning/moveit/blob/master/moveit_commander/src/moveit_commander/move_group.py
"""
def plan_pick_and_place_4Step_PickNPlace(req):
    # print("Message received from Unity: {}, {}, {}".format(req.pick_pose, req.robot_joints.joints, req.robotArmRotation))
    response = MoverService_PickNPlaceResponse()

    current_robot_joint_configuration = req.robot_joints.joints

    # Pre grasp - position gripper directly above target object
    pre_grasp_pose = plan_trajectory(move_group, req.pick_pose, current_robot_joint_configuration, req.robotArmRotation)
    
    # If the trajectory has no points, planning has failed and we return an empty response
    if not pre_grasp_pose.joint_trajectory.points:
        response.response_msg = "Target is out of reach"
        return response

    previous_ending_joint_angles = pre_grasp_pose.joint_trajectory.points[-1].positions

    # Grasp - lower gripper so that fingers are on either side of object
    pick_pose = copy.deepcopy(req.pick_pose)
    pick_pose.position.z -= 0.05  # Static value coming from Unity, TODO: pass along with request
    grasp_pose = plan_trajectory(move_group, pick_pose, previous_ending_joint_angles, req.robotArmRotation)
    
    if not grasp_pose.joint_trajectory.points:
        response.response_msg = "Target Z-position is out of reach"
        return response

    previous_ending_joint_angles = grasp_pose.joint_trajectory.points[-1].positions

    # Pick Up - raise gripper back to the pre grasp position
    pick_up_pose = plan_trajectory(move_group, req.pick_pose, previous_ending_joint_angles, req.robotArmRotation)
    
    if not pick_up_pose.joint_trajectory.points:
        response.response_msg = "Failed to return back to Pre-Grasp position"
        return response

    previous_ending_joint_angles = pick_up_pose.joint_trajectory.points[-1].positions

    # Place - move gripper to desired placement position
    place_pose = plan_trajectory(move_group, req.place_pose, previous_ending_joint_angles, req.robotArmRotation)

    if not place_pose.joint_trajectory.points:
        response.response_msg = "Placement is out of reach"
        return response

    # If trajectory planning worked for all pick and place stages, add plan to response
    response.trajectories.append(pre_grasp_pose)
    response.trajectories.append(grasp_pose)
    response.trajectories.append(pick_up_pose)
    response.trajectories.append(place_pose)

    move_group.clear_pose_targets()

    return response


def plan_UsingPlanningSequencer(req):
    print("Message received from Unity: {}, {}, {}, {}".format(req.target_poses, req.initial_robot_joints.joints, req.robot_joints_per_sequence, req.robotArmRotation))
    response = MoverService_UsingPlanningSequencerResponse()

    current_robot_joint_configuration = req.initial_robot_joints.joints

    for i in range(len(req.target_poses)):
        target_pose = req.target_poses[i]
        pose = plan_trajectory(move_group, target_pose, current_robot_joint_configuration, req.robotArmRotation)

        # If the trajectory has no points, planning has failed and we return an empty response
        if not pose.joint_trajectory.points:
            print("Failed to plan for target_pose : {}".format(i+1))
            continue  
            # return response

        print("Successful plan! for target_pose : {}".format(i+1))
        current_robot_joint_configuration = req.robot_joints_per_sequence[i].joints
        response.trajectories.append(pose)

    move_group.clear_pose_targets()
    return response



# Initialize global variables
planning = False
motionPlan = None
lastValidTargetPose = None

def fanuc_m20ib25_unity_callback(msg):
    global planning, motionPlan, lastValidTargetPose
    if not planning:
        planning = True
        # print("Message received from Unity: {}, {}, {}".format(msg.target_pose, msg.robot_joints.joints, msg.robotArmRotation))
        motionPlan = plan_trajectory(move_group, msg.target_pose, msg.robot_joints.joints, msg.robotArmRotation)
        lastValidTargetPose = msg.target_pose
        # Clear targets for the next iteration
        move_group.clear_pose_targets()
    

def moveit_server():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('fanuc_m20ib25_server')

    group_name = "fanuc_m20ib25"
    global move_group
    move_group = moveit_commander.MoveGroupCommander(group_name)
    move_group.set_pose_reference_frame('base_link')  # Assuming 'base_link' is the robot's base frame

    global planning, motionPlan, lastValidTargetPose
    planning = False
    motionPlan = None
    lastValidTargetPose = None

    global pub
    pub = rospy.Publisher('fanuc_m20ib25_ROS_joints_to_unity', FanucM20iB25ROSJointsToUnity, queue_size=10)
    rospy.Subscriber('fanuc_m20ib25_unity_target_pose_to_ROS', FanucM20iB25UnityTargetPoseToROS, fanuc_m20ib25_unity_callback)
    service = rospy.Service('fanuc_m20ib25_msgs_4Step_PickNPlace', MoverService_PickNPlace, plan_pick_and_place_4Step_PickNPlace)
    service2 = rospy.Service('fanuc_m20ib25_msgs_PlanningSequencer', MoverService_UsingPlanningSequencer, plan_UsingPlanningSequencer)

    print("Ready to plan")

    rate = rospy.Rate(10)  # 10 Hz loop rate (adjust as needed)
    while not rospy.is_shutdown():
        # Publish after a request has been made 
        if motionPlan:
            response_msg = FanucM20iB25ROSJointsToUnity()

            if motionPlan.joint_trajectory.points:
                response_msg.trajectory = motionPlan
                response_msg.last_valid_target_pose = lastValidTargetPose
                pub.publish(response_msg)
                # print("Sending Motion Plan to Unity")
                # # Execute the planned motion
                # move_group.execute(motionPlan, wait=True)
            else:
                pub.publish(response_msg)
                # print("Trajectory could not be planned")
            
            # # Clear targets for the next iteration
            # move_group.clear_pose_targets()
            planning = False
            motionPlan = None
            lastValidTargetPose = None

        # Sleep to maintain loop rate
        rate.sleep()

    # Shutdown MoveIt! commander
    moveit_commander.roscpp_shutdown()

    # rospy.spin()


if __name__ == "__main__":
    moveit_server()