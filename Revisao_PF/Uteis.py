#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy

import numpy as np
import cv2
from geometry_msgs.msg import Twist, PointStamped, Point, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge,CvBridgeError
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import biblioteca_cow

""" 
Running Garra
    roslaunch mybot_description mybot_control2.launch
"""

# Antes de começar, copie a pasta "MobileNet" para o seu diretório, assim como os arquivos "aruco_3d.py" e "biblioteca_cow.py"

class Control():
    def __init__(self):
        self.rate = rospy.Rate(250) # 250 Hz
        
        self.robot_state = "procura"
        self.robot_machine = {
            "procura": self.procura,
            "aproxima": self.aproxima,
            "stop": self.stop,
            "aproxima_2": self.aproxima_2,
            "rotate" : self.rotate,
            "aproxima_amarelo": self.aproxima_amarelo,
        }
        self.point = Point(x=0, y=0, z=0)
        self.midle = -1
        self.kp = 0.05

        self.x_green = 0
        self.y_green = 0

        self.x_yellow = 0
        self.y_yellow = 0

        self.x_red = 0
        self.y_red = 0

        self.x_white = 0
        self.y_white = 0

        self.ids = 0
        self.x_arucos = 0
        self.distances_arucos = 0
        self.countours_green = 0
        self.countours_white = 0
        self.countours_yellow = 0
        self.max_area_green = 0
        self.max_area_red = 0
        self.max_area_blue = 0

        # MobileNet
        self.init_m = True
        self.net = biblioteca_cow.load_mobilenet()
        self.confidance = 0.5
        self.results = None
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
        self.CONFIDENCE = 0.7
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        self.object = None
        self.prob = 0
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.center_x = 0
        self.center_y = 0

        # Subscribers
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
        self.image_sub_mobile = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback_mobile, queue_size=1, buff_size = 2**24)
        self.odom_sub = rospy.Subscriber("/odom",Odometry,self.odom_callback)
        self.infos_arucos_sub = rospy.Subscriber("/aruco_info", PoseArray, self.aruco_callback, queue_size=1)
        self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)

        # Publishers
        self.point_pub = rospy.Publisher('/center_publisher', PointStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.arm_pub = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
        self.clamp_pub = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)
        
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

        # Blue
        self.lower_hsv_blue = np.array([113,148,232],dtype=np.uint8)
        self.upper_hsv_blue = np.array([134,255,255],dtype=np.uint8)

        self.kernel = np.ones((5,5),np.uint8)

        self.arm_pub.publish(-1)
        self.clamp_pub.publish(-1)

    # Controle do Braço
    def down(self):
        self.arm_pub.publish(-1)
    def front(self):
        self.arm_pub.publish(0)
    def up(self):
        self.arm_pub.publish(1.5)
    
    # Controle da Garra
    def open(self):
        self.clamp_pub.publish(-1)
    def close(self):
        self.clamp_pub.publish(0)

    def laser_callback(self, msg: LaserScan) -> None:
        """
        Callback function for the laser topic
        """
        self.laser_msg = np.array(msg.ranges).round(decimals=2) # Converte para np.array e arredonda para 2 casas decimais
        self.laser_msg[self.laser_msg == 0]

        self.laser_forward = list(self.laser_msg[:5]) + list(self.laser_msg[-5:])
        self.laser_backwards = list(self.laser_msg[175:180]) + list(self.laser_msg[180:185])

        print(f'A distanciado lidar é exatamente essa: {np.min(self.laser_forward)}')

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

    def image_callback_mobile(self, msg: CompressedImage) -> None:
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.midle = cv_image.shape[1]//2
        except CvBridgeError as e:
            print(e)
        if self.init_m is True:
            _, self.results = biblioteca_cow.detect(self.net, cv_image, self.CONFIDENCE, self.COLORS, self.CLASSES)
            self.object = self.results[0][0]
            self.prob = self.results[0][1]
            self.x1 = self.results[0][2][0]
            self.y1 = self.results[0][2][1]
            self.x2 = self.results[0][3][0]
            self.y2 = self.results[0][3][1]

            self.center_x = (self.x1 + self.x2) / 2
            self.center_y = (self.y1 + self.y2) / 2

            print(f'x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}')

    def aruco_callback(self, msg: PoseArray) -> None:
        """
        Callback function for the aruco topic
        """
        if len(msg.poses) > 0:
            self.ids=msg.poses[0].position.x
            self.distances_arucos=msg.poses[0].position.y
            self.x_arucos=msg.poses[0].position.z
            
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

        if len(self.countours_white) > 0:
            c = max(self.countours_white, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                self.x_white = int(M["m10"] / M["m00"])
                self.y_white = int(M["m01"] / M["m00"])

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
                        #cv2.circle(bgr, (self.x_green, self.y_green), 5, (0, 0, 255), -1)

        # MÁSCARA VERMELHA
        mask_red = cv2.inRange(hsv, self.lower_hsv_red, self.upper_hsv_red)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, self.kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, self.kernel)

        countours_red,_ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(countours_red) > 0:
            self.max_area_red = 0
            for c in countours_red:
                area = cv2.contourArea(c)
                if area > self.max_area_red:
                    self.max_area_red = area
                    M = cv2.moments(c)
                    self.x_red = int(M["m10"] / M["m00"])
                    self.y_red = int(M["m01"] / M["m00"])
                    #cv2.circle(bgr, (self.x_red, self.y_red), 5, (255, 0, 0), -1)

        # MÁSCARA AZUL
        mask_blue = cv2.inRange(hsv, self.lower_hsv_blue, self.upper_hsv_blue)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.kernel)

        self.countours_blue,_ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(self.countours_blue) > 0:
            self.max_area_blue = 0
            for c in self.countours_blue:
                area = cv2.contourArea(c)
                if area > self.max_area_blue:
                    self.max_area_blue = area
                    M = cv2.moments(c)
                    self.x_blue = int(M["m10"] / M["m00"])
                    self.y_blue = int(M["m01"] / M["m00"])
                    #cv2.circle(bgr, (self.x_blue, self.y_blue), 5, (0, 0, 255), -1)
                    #print(f'A Area azul é exatamanete essa {self.max_area_blue}')

        cv2.imshow("Image window", bgr)
        cv2.waitKey(1)

    def aproxima_amarelo(self) -> None:
        """
        Aproxima do amarelo
        """
        self.twist.linear.x = 0.15
        error = self.midle - self.x_yellow
        self.twist.angular.z = float(error)/1000
        

    def procura(self) -> None:
        """
        Find countours
        """
        self.twist.angular.z = 0.1
        self.ids = 0
        self.max_area_green = 0
        print(self.yaw)
        if self.object == 'bicycle' and self.prob > 80 and (self.midle - self.center_x < 10):
            self.robot_state = "aproxima"

    def aproxima(self) -> None:
        self.twist.linear.x = 0.15
        error = self.midle - self.center_x
        self.twist.angular.z = float(error)/1000

        if np.min(self.laser_forward) < 0.3:
            self.robot_state = "rotate"
            
    def rotate(self) -> None:
        self.twist.angular.z = 0.2
        if self.yaw > 2.295 + np.pi - 0.1 and  self.yaw < 2.295 + np.pi + 0.1:
           self.robot_state = "aproxima_2"

    def aproxima_2(self) -> None:
        self.twist.linear.x = 0.1
        if np.min(self.laser_forward) < 0.3:
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

# Vídeo Simulador: 
# Vídeo Real: 