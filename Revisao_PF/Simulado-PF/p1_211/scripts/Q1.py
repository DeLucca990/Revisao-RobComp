#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy

import numpy as np
import cv2
from geometry_msgs.msg import Twist, PointStamped, Point, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge,CvBridgeError
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

""" 
Running
    roslaunch my_simulation pista23-1.launch
    rosrun aps4 pista.py
"""

class Control():
    def __init__(self):
        self.rate = rospy.Rate(250) # 250 Hz
        
        self.robot_state = "aproxima"
        self.robot_machine = {
            "aproxima": self.aproxima,
            "first_turn": self.first_turn,
            "stop": self.stop
        }
        self.point = Point(x=0, y=0, z=0)
        self.midle = -1
        self.kp = 0.05

        self.x0 = 0
        self.y0 = 0

        self.x_green = 0
        self.y_green = 0

        self.x_yellow = 0
        self.y_yellow = 0

        self.ids = 0
        self.distances_arucos = 0
        self.countours_green = 0
        self.countours_white = 0
        self.countours_yellow = 0
        self.max_area_green = 0
        self.max_area_red = 0

        # Subscribers
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
        self.odom_sub = rospy.Subscriber("/odom",Odometry,self.odom_callback)
        self.infos_arucos_sub = rospy.Subscriber("/aruco_info", PoseArray, self.aruco_callback, queue_size=1)
        self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

        # Publishers
        self.point_pub = rospy.Publisher('/center_publisher', PointStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # HSV Filter
        # Yellow
        self.lower_hsv = np.array([14,200,233],dtype=np.uint8)
        self.upper_hsv = np.array([49,255,255],dtype=np.uint8)
        
        # Green
        self.lower_hsv_green = np.array([41,72,104],dtype=np.uint8)
        self.upper_hsv_green = np.array([100,255,255],dtype=np.uint8)

        # Red
        self.lower_hsv_red = np.array([0,212,111],dtype=np.uint8)
        self.upper_hsv_red = np.array([11,255,255],dtype=np.uint8)

        # White
        self.lower_hsv_white = np.array([0,0,236],dtype=np.uint8)
        self.upper_hsv_white = np.array([179,74,255],dtype=np.uint8)

        self.kernel = np.ones((5,5),np.uint8)

    def laser_callback(self, msg: LaserScan) -> None:
        """
        Callback function for the laser topic
        """
        self.laser_msg = np.array(msg.ranges).round(decimals=2) # Converte para np.array e arredonda para 2 casas decimais
        self.laser_msg[self.laser_msg == 0]

        self.laser_forward = list(self.laser_msg[:5]) + list(self.laser_msg[-5:])
        self.laser_backwards = list(self.laser_msg[175:180]) + list(self.laser_msg[180:185])

    def odom_callback(self, data: Odometry):
        self.odom = data
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z
		
        orientation_list = [data.pose.pose.orientation.x,
							data.pose.pose.orientation.y,
							data.pose.pose.orientation.z,
							data.pose.pose.orientation.w]

        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

        self.yaw = self.yaw % (2*np.pi)

    def image_callback(self, msg: CompressedImage) -> None:
        """
        Callback function for the image topic
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.midle = cv_image.shape[1]//2
        except CvBridgeError as e:
            print(e)
        
        self.color_segmentation(cv_image)

    def aruco_callback(self, msg: PoseArray) -> None:
        """
        Callback function for the aruco topic
        """
        if len(msg.poses) > 0:
            self.ids=msg.poses[0].position.x
            self.distances_arucos=msg.poses[0].position.y
            
    def color_segmentation(self,bgr: np.ndarray) -> None:
        """ 
        Use HSV color space to segment the image and find the center of the object.

        Args:
            bgr (np.ndarray): image in BGR format
        """
        # MÁSCARA BRANCA
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, self.lower_hsv_white, self.upper_hsv_white)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, self.kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, self.kernel)

        self.countours_white,_ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # MÁSCARA AMARELA
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        self.countours_yellow,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(self.countours_yellow) > 0:
            c = max(self.countours_yellow, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                self.x_yellow = int(M["m10"] / M["m00"])
                self.y_yellow = int(M["m01"] / M["m00"])
                cv2.circle(bgr, (self.x_yellow, self.y_yellow), 5, (0, 0, 255), -1)
        else:
            self.x_yellow = -1

        # MÁSCARA VERDE
        mask_green = cv2.inRange(hsv, self.lower_hsv_green, self.upper_hsv_green)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, self.kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, self.kernel)

        self.countours_green,_ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(self.countours_green) > 0:
            self.max_area_green = 0
            for c in self.countours_green:
                area = cv2.contourArea(c)
                if area > self.max_area_green:
                    self.max_area_green = area
                    if self.max_area_green > 350:
                        M = cv2.moments(c)
                        self.x_green = int(M["m10"] / M["m00"])
                        self.y_green = int(M["m01"] / M["m00"])
                        cv2.circle(bgr, (self.x_green, self.y_green), 5, (0, 0, 255), -1)
                        print(self.max_area_green)

        # MÁSCARA VERMELHA

        cv2.imshow("Image window", bgr)
        cv2.waitKey(1)

    def aproxima(self) -> None:
        self.twist.linear.x = 0.1
        error = self.midle - self.x_yellow
        self.twist.angular.z = float(error)/100
        print(self.yaw)
        if (0.3 > self.yaw or self.yaw > 6) or (self.yaw > np.pi/2 - 0.3 and self.yaw < np.pi/2 + 0.3):
            self.twist.linear.x = 0.5
        if self.max_area_green > 60000:
            self.robot_state =  "first_turn"


    def first_turn(self) -> None:
        self.twist.linear.x = 0.17
        self.twist.angular.z = 0.12
        if np.min(self.laser_forward) < 0.5:
            self.robot_state = "stop"

    def stop(self) -> None:
        self.twist = Twist()

    def control(self) -> None:
        '''
        This function is called at least at {self.rate} Hz.
        This function controls the robot.
        Não modifique esta função.
        '''
        self.twist = Twist()
        print(f'self.robot_state: {self.robot_state}')
        self.robot_machine[self.robot_state]()

        self.cmd_vel_pub.publish(self.twist)
        
        self.rate.sleep()

def main():
    rospy.init_node('Controler')
    control = Control()
    rospy.sleep(1)

    while not rospy.is_shutdown():
        control.control()

if __name__=="__main__":
    main()