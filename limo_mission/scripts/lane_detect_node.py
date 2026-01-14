#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Detection Node for LIMO Pro
- Detects white lane lines using HLS colorspace
- Calculates steering offset based on lane center
- Publishes visualization via ROS topics
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Int32, Bool, String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from limo_mission.cfg import LaneDetectConfig


class LaneDetectNode:
    def __init__(self):
        rospy.init_node('lane_detect_node')
        
        self.bridge = CvBridge()
        
        # Default parameters (WHITE lanes - HLS)
        self.LANE_LOW = np.array([0, 180, 0])
        self.LANE_HIGH = np.array([180, 255, 60])
        
        # ROI settings
        self.roi_top = rospy.get_param('~roi_top', 300)
        self.roi_bottom = rospy.get_param('~roi_bottom', 450)
        
        # Control parameters
        self.base_speed = rospy.get_param('~base_speed', 0.3)
        self.kp = rospy.get_param('~kp', 0.01)
        self.kd = rospy.get_param('~kd', 0.005)
        self.max_steering = rospy.get_param('~max_steering', 0.5)
        self.lane_offset = rospy.get_param('~lane_offset', 180)
        
        # cmd_vel publishing option (for standalone lane tracing)
        self.publish_cmd_vel = rospy.get_param('~publish_cmd_vel', False)
        
        # Stop flag (from traffic light or other nodes)
        self.stop_flag = False
        
        # Parking flag (주차 중일 때 제어권 양보)
        self.parking_active = False
        
        # PD control state
        self.prev_offset = 0
        
        # Image dimensions
        self.img_width = 640
        self.img_height = 480
        self.img_center = self.img_width // 2
        
        # Publishers
        self.pub_image = rospy.Publisher('/limo/lane_detect/image', Image, queue_size=1)
        self.pub_steering = rospy.Publisher('/limo/steering_offset', Float32, queue_size=1)
        self.pub_center_x = rospy.Publisher('/limo/lane_center_x', Int32, queue_size=1)
        self.pub_speed = rospy.Publisher('/limo/lane_speed', Float32, queue_size=1)
        
        # cmd_vel publisher (only when publish_cmd_vel is True)
        if self.publish_cmd_vel:
            self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            # Subscribe to stop flag from traffic light
            self.sub_stop = rospy.Subscriber('/limo/traffic_stop', Bool, self.stop_callback, queue_size=1)
            # Subscribe to parking state
            self.sub_parking = rospy.Subscriber('/parking/state', String, self.parking_callback, queue_size=1)
        
        # Subscriber
        self.sub_image = rospy.Subscriber(
            '/camera/color/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # Dynamic reconfigure
        self.srv = Server(LaneDetectConfig, self.reconfigure_callback)
        
        rospy.loginfo("="*50)
        rospy.loginfo("Lane Detect Node initialized")
        rospy.loginfo(f"kp={self.kp}, kd={self.kd}, lane_offset={self.lane_offset}")
        rospy.loginfo(f"publish_cmd_vel: {self.publish_cmd_vel}")
        rospy.loginfo("View: rqt_image_view /limo/lane_detect/image")
        rospy.loginfo("="*50)
    
    def reconfigure_callback(self, config, level):
        """Dynamic reconfigure callback"""
        self.LANE_LOW = np.array([config.lane_h_low, config.lane_l_low, config.lane_s_low])
        self.LANE_HIGH = np.array([config.lane_h_high, config.lane_l_high, config.lane_s_high])
        self.roi_top = config.roi_top
        self.roi_bottom = config.roi_bottom
        self.base_speed = config.base_speed
        self.kp = config.kp
        self.kd = config.kd
        self.max_steering = config.max_steering
        self.lane_offset = config.lane_offset
        
        rospy.loginfo(f"Updated: kp={self.kp}, kd={self.kd}, lane_offset={self.lane_offset}")
        return config
    
    def stop_callback(self, msg):
        """Stop flag callback from traffic light"""
        self.stop_flag = msg.data
    
    def parking_callback(self, msg):
        """Parking state callback - 주차 진행 중이면 제어권 양보, DONE이면 완전 중지"""
        # IDLE 상태일 때만 lane_detect가 제어권 가짐
        # DONE 상태면 주차 완료 - 라인 트레이싱도 종료
        if msg.data == "DONE":
            self.parking_active = True  # 영구 중지
            rospy.loginfo("[lane_detect] Parking complete - lane tracing stopped")
        else:
            self.parking_active = (msg.data != "IDLE")
    
    def detect_lanes(self, img):
        """Detect lane lines in ROI"""
        roi = img[self.roi_top:self.roi_bottom, :]
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        
        # Create mask for lane color
        mask = cv2.inRange(hls, self.LANE_LOW, self.LANE_HIGH)
        
        # Morphology to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Split into left and right regions
        mid = roi.shape[1] // 2
        left_mask = mask[:, :mid]
        right_mask = mask[:, mid:]
        
        left_x, right_x = -1, -1
        
        # Left lane
        M_left = cv2.moments(left_mask)
        if M_left['m00'] > 0:
            left_x = int(M_left['m10'] / M_left['m00'])
        
        # Right lane
        M_right = cv2.moments(right_mask)
        if M_right['m00'] > 0:
            right_x = int(M_right['m10'] / M_right['m00']) + mid
        
        # Calculate lane center
        if left_x > 0 and right_x > 0:
            lane_center_x = (left_x + right_x) // 2
        elif left_x > 0:
            lane_center_x = left_x + self.lane_offset
        elif right_x > 0:
            lane_center_x = right_x - self.lane_offset
        else:
            lane_center_x = mid
        
        return lane_center_x, left_x, right_x, mask
    
    def image_callback(self, msg):
        """Process incoming image"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return
            
            self.img_height, self.img_width = img.shape[:2]
            self.img_center = self.img_width // 2
            
            # Detect lanes
            lane_center_x, left_x, right_x, mask = self.detect_lanes(img)
            
            # Calculate error and PD control
            error = lane_center_x - self.img_center
            derivative = error - self.prev_offset
            steering = np.clip(
                self.kp * error + self.kd * derivative,
                -self.max_steering, self.max_steering
            )
            self.prev_offset = error
            
            # Publish results
            self.pub_center_x.publish(Int32(data=lane_center_x))
            self.pub_steering.publish(Float32(data=steering))
            self.pub_speed.publish(Float32(data=self.base_speed))  # 다른 노드들이 사용
            
            # Publish cmd_vel if enabled (주차 중이면 발행 안함)
            if self.publish_cmd_vel and not self.parking_active:
                cmd = Twist()
                if self.stop_flag:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                else:
                    cmd.linear.x = self.base_speed
                    cmd.angular.z = -steering
                self.pub_cmd.publish(cmd)
            
            # Visualization
            vis_img = img.copy()
            roi_y = (self.roi_top + self.roi_bottom) // 2
            
            # Draw ROI
            cv2.rectangle(vis_img, (0, self.roi_top), (self.img_width, self.roi_bottom), (255, 255, 0), 2)
            
            # Draw lane points
            if left_x > 0:
                cv2.circle(vis_img, (left_x, roi_y), 10, (255, 0, 0), -1)
            if right_x > 0:
                cv2.circle(vis_img, (right_x, roi_y), 10, (0, 0, 255), -1)
            
            # Draw lane center and image center
            cv2.circle(vis_img, (lane_center_x, roi_y), 10, (0, 255, 0), -1)
            cv2.line(vis_img, (self.img_center, self.roi_top), (self.img_center, self.roi_bottom), (0, 255, 255), 2)
            cv2.arrowedLine(vis_img, (self.img_center, roi_y), (lane_center_x, roi_y), (0, 255, 0), 3)
            
            # Info text
            cv2.putText(vis_img, f"Offset: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_img, f"Steering: {steering:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_img, f"L:{left_x} R:{right_x}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Publish image
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Lane detect error: {e}")
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = LaneDetectNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
