#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# 
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Robot:
    """ Robot class used for taking in location from a camera or video and centering it in the frame. """
    
    def __init__(self, tb_vel, camera_resolution):
        """ Initializer for the class. Sets some needed ros pubs and subs along with velocity and padding for the
            centering function. """
        self.robot_home_pos = self.roll = self.pitch = self.yaw = 0.0        
        self.ROS_RATE = 70                          # hz to send command to robot. default: 10
        self.ROTATE_VELOCITY = tb_vel
        self.RETURN_HOME_VELOCITY = 0.65            # velocity for automoving back to home position. default: 0.65
        self.CAM_RESOLUTION = camera_resolution
        self.PADDING_FACTOR = 8.0                   # padding +/- center of screen for centering object. 
                                                    # bigger number = smaller area. default: 8.0 
                
        # initiliaze the turtlebot node name
        rospy.init_node('turtlebot_finder', anonymous = False)
        
        # subscribe to the odometry node
        rospy.Subscriber('odom', Odometry, self.determine_robot_pose)
        
        # tell ros what function to call when the robot is shutdown    
        rospy.on_shutdown(self.ros_shutdown)

        # create a publisher which can "talk" to TurtleBot and tell it to move
        self.ros_publisher = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)

        # set the command rate for the robot (how often should we send commands to the robot?)
        self.send_cmd_rate = rospy.Rate(self.ROS_RATE);

        # twist is a datatype for velocity
        self.robot_move_cmd = Twist()
        

    def determine_robot_pose(self, msg):
        """ Determines the robot's position """
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion (orientation_list)
        
        
    def get_current_pos(self):
        """ Get the robot's current position. """ 
        robot_pos = quaternion_from_euler(self.roll, self.pitch, self.yaw)
        return robot_pos
    
    
    def set_home_pos(self):
        """ Set the robot's home location. """
        self.robot_home_pos = self.get_current_pos()

        
    def turtlebot_stop(self):
        """ Returns the velocity for stopping the robot. """
        return 0.0


    def turtlebot_right(self):
        """ Returns the velocity for moving to the right. """
        return (self.ROTATE_VELOCITY)


    def turtlebot_left(self):
        """ Returns the velocity for moving to the left. """
        return -(self.ROTATE_VELOCITY)


    def turtlebot_automove_left(self):
        """ Returns the velocity for auto-moving to the left. """
        return -(self.RETURN_HOME_VELOCITY)


    def turtlebot_automove_right(self):
        """ Returns the velocity for auto-moving to the right. """
        return (self.RETURN_HOME_VELOCITY)
        
    def get_vel(self):
        """ Returns the robot's current velocity """
        return self.robot_move_cmd.angular.z
        
    def determine_robot_action(self, obj_x_coord):
        """ Determine what the robot should do based on the x coordinate of the object. """
        # define the center of the captured image and determine detection padding 
        center_of_image = int(self.CAM_RESOLUTION[0]/2.0)
        padding = int(self.CAM_RESOLUTION[0]/self.PADDING_FACTOR)
        
        ### determine if robot should stop, rotate left, or rotate right ###
        
        # DETECT NOTHING - return to home, auto-rotate 
        if (obj_x_coord == -1):
            current_pos = self.get_current_pos()
            # print("home: " + str(self.robot_home_pos[2]) + " current pos: " + str(current_pos[2]))
            if (current_pos[2] > self.robot_home_pos[2] + 0.05):
                self.robot_move_cmd.angular.z = self.turtlebot_automove_left()
                return
            elif (current_pos[2] < self.robot_home_pos[2] - 0.05):
                self.robot_move_cmd.angular.z = self.turtlebot_automove_right()
                return
            else:
                self.robot_move_cmd.angular.z = self.turtlebot_stop()
                return
                
        # DETECTED AN OBJECT - try to center robot on object
        elif (obj_x_coord > (center_of_image + padding) and obj_x_coord < self.CAM_RESOLUTION[0]):
            self.robot_move_cmd.angular.z = self.turtlebot_right()
            return
        elif (obj_x_coord < (center_of_image - padding) and obj_x_coord > 0):
            self.robot_move_cmd.angular.z = self.turtlebot_left()
            return
        elif (obj_x_coord > 0):
            self.robot_move_cmd.angular.z = self.turtlebot_stop()
            return

    def take_robot_action(self):
        """ Publish the command to ros master. """    
        self.ros_publisher.publish(self.robot_move_cmd)
        self.send_cmd_rate.sleep()
        
        
    def ros_shutdown(self):
        """ Show this message and log when robot shuts down """
        rospy.loginfo("Shutting down...")
        
        
                 
