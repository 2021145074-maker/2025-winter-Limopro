#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Light Detection Node for LIMO Pro
- Detects red and green traffic lights using HSV colorspace
- Uses rectangularity filtering to identify rectangular signs
- Publishes visualization via ROS topics (no cv2.imshow)
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from limo_mission.cfg import TrafficLightConfig


class TrafficLightNode:
    def __init__(self):
        rospy.init_node('traffic_light_node')
        
        self.bridge = CvBridge()
        
        # Enable flag for sequence control
        self.enabled = True
        self.green_passed = False  # Flag to signal green light passed
        
        # Red HSV thresholds (two ranges for hue wrap-around)
        self.RED_LOW1 = np.array([0, 100, 100])
        self.RED_HIGH1 = np.array([10, 255, 255])
        self.RED_LOW2 = np.array([160, 100, 100])
        self.RED_HIGH2 = np.array([180, 255, 255])
        
        # Green HSV thresholds
        self.GREEN_LOW = np.array([35, 100, 100])
        self.GREEN_HIGH = np.array([85, 255, 255])
        
        # Detection parameters
        self.min_area = 500
        self.circularity_thresh = 0.6
        
        # ROI settings (신호등은 화면 상단)
        self.roi_top = rospy.get_param('~roi_top', 0)
        self.roi_bottom = rospy.get_param('~roi_bottom', 240)
        self.roi_left = rospy.get_param('~roi_left', 0)
        self.roi_right = rospy.get_param('~roi_right', 640)
        
        # Brightness adjustment
        self.brightness_factor = rospy.get_param('~brightness_factor', 0.5)
        
        # Lane tracing parameters (from lane_detect_node)
        self.lane_speed = 0.25  # 기본값, lane_detect_node에서 수신
        
        # Lane steering from lane_detect_node
        self.lane_steering = 0.0
        
        # State
        self.current_light = "NONE"
        self.stop_flag = False
        
        # Publishers - Image for rqt_image_view
        self.pub_image = rospy.Publisher('/limo/traffic_light/image', Image, queue_size=1)
        self.pub_debug = rospy.Publisher('/limo/traffic_light/debug', Image, queue_size=1)
        # CompressedImage for bandwidth
        self.pub_image_compressed = rospy.Publisher('/limo/traffic_light/image/compressed', CompressedImage, queue_size=1)
        self.pub_debug_compressed = rospy.Publisher('/limo/traffic_light/debug/compressed', CompressedImage, queue_size=1)
        self.pub_light = rospy.Publisher('/limo/traffic_light', String, queue_size=1)
        self.pub_stop = rospy.Publisher('/limo/traffic_stop', Bool, queue_size=1)
        self.pub_passed = rospy.Publisher('/limo/traffic_passed', Bool, queue_size=1)
        
        # cmd_vel publisher for direct control
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscriber
        self.sub_image = rospy.Subscriber(
            '/camera/color/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # Steering and speed subscriber from lane_detect_node
        self.sub_steering = rospy.Subscriber('/limo/steering_offset', Float32, self.steering_callback, queue_size=1)
        self.sub_speed = rospy.Subscriber('/limo/lane_speed', Float32, self.speed_callback, queue_size=1)
        
        # Enable subscriber for sequence control
        self.sub_enable = rospy.Subscriber('/limo/traffic_enable', Bool, self.enable_callback, queue_size=1)
        
        # Dynamic reconfigure
        self.srv = Server(TrafficLightConfig, self.reconfigure_callback)
        
        # Control loop timer
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        
        rospy.loginfo("="*50)
        rospy.loginfo("Traffic Light Node initialized")
        rospy.loginfo(f"lane_speed: {self.lane_speed} (from lane_detect_node)")
        rospy.loginfo("View: rqt_image_view /limo/traffic_light/image")
        rospy.loginfo("="*50)
    
    def steering_callback(self, msg):
        """Steering callback from lane_detect_node"""
        self.lane_steering = msg.data
    
    def speed_callback(self, msg):
        """Speed callback from lane_detect_node"""
        self.lane_speed = msg.data
    
    def enable_callback(self, msg):
        """Enable/disable callback for sequence control"""
        self.enabled = msg.data
        if self.enabled:
            # Reset passed flag when re-enabled
            self.green_passed = False
        rospy.loginfo(f"Traffic light detect {'ENABLED' if self.enabled else 'DISABLED'}")
    
    def reconfigure_callback(self, config, level):
        """Dynamic reconfigure callback"""
        self.RED_LOW1 = np.array([config.red_h_low1, config.red_s_low, config.red_v_low])
        self.RED_HIGH1 = np.array([config.red_h_high1, config.red_s_high, config.red_v_high])
        self.RED_LOW2 = np.array([config.red_h_low2, config.red_s_low, config.red_v_low])
        self.RED_HIGH2 = np.array([config.red_h_high2, config.red_s_high, config.red_v_high])
        
        self.GREEN_LOW = np.array([config.green_h_low, config.green_s_low, config.green_v_low])
        self.GREEN_HIGH = np.array([config.green_h_high, config.green_s_high, config.green_v_high])
        
        self.min_area = config.min_area
        self.circularity_thresh = config.circularity_thresh
        
        # ROI settings
        self.roi_top = config.roi_top
        self.roi_bottom = config.roi_bottom
        self.roi_left = config.roi_left
        self.roi_right = config.roi_right
        
        # Brightness
        self.brightness_factor = config.brightness_factor
        
        rospy.loginfo(f"Updated: ROI=[{self.roi_top}:{self.roi_bottom}]")
        return config
    
    def control_loop(self, event):
        """Control loop - publish cmd_vel based on traffic light state"""
        cmd = Twist()
        
        if self.stop_flag:
            # RED light - stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # GREEN or no light - lane tracing (lane_detect_node의 속도 사용)
            cmd.linear.x = self.lane_speed
            cmd.angular.z = -self.lane_steering
        
        self.pub_cmd.publish(cmd)
    
    def publish_image_raw(self, publisher, img):
        """Publish image as raw Image"""
        try:
            msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            msg.header.stamp = rospy.Time.now()
            publisher.publish(msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish raw image: {e}")
    
    def publish_image_compressed(self, publisher, img):
        """Publish image as CompressedImage"""
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        publisher.publish(msg)
    
    def detect_color(self, hsv, low1, high1, low2=None, high2=None):
        """Detect color with optional second range (for red)"""
        mask1 = cv2.inRange(hsv, low1, high1)
        if low2 is not None and high2 is not None:
            mask2 = cv2.inRange(hsv, low2, high2)
            return cv2.bitwise_or(mask1, mask2)
        return mask1
    
    def find_rectangular_contours(self, mask, color_name):
        """Find rectangular contours in mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            if rect_area == 0:
                continue
            
            # Rectangularity = contour area / bounding rect area
            rectangularity = area / rect_area
            
            if rectangularity >= self.circularity_thresh:
                rectangles.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'center': (x + w // 2, y + h // 2),
                    'area': area,
                    'rectangularity': rectangularity,
                    'color': color_name
                })
        
        return rectangles
    
    def image_callback(self, msg):
        """Image callback"""
        # Skip processing if disabled
        if not self.enabled:
            return
            
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            full_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if full_img is None:
                return
            
            # Extract ROI (신호등 영역만)
            h, w = full_img.shape[:2]
            roi_top = max(0, self.roi_top)
            roi_bottom = min(h, self.roi_bottom)
            roi_left = max(0, self.roi_left)
            roi_right = min(w, self.roi_right)
            
            roi_original = full_img[roi_top:roi_bottom, roi_left:roi_right].copy()
            
            # Brightness adjustment (밝은 환경에서 ROI만 어둡게)
            if self.brightness_factor < 1.0:
                img = cv2.convertScaleAbs(roi_original, alpha=self.brightness_factor, beta=0)
            else:
                img = roi_original.copy()
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Create masks
            red_mask = self.detect_color(hsv, self.RED_LOW1, self.RED_HIGH1, self.RED_LOW2, self.RED_HIGH2)
            green_mask = self.detect_color(hsv, self.GREEN_LOW, self.GREEN_HIGH)
            
            # Find rectangular contours
            red_rects = self.find_rectangular_contours(red_mask, "RED")
            green_rects = self.find_rectangular_contours(green_mask, "GREEN")
            
            # Determine traffic light state
            prev_light = self.current_light
            
            if red_rects:
                self.current_light = "RED"
                self.stop_flag = True
            elif green_rects:
                self.current_light = "GREEN"
                self.stop_flag = False
                # Signal green light passed (only once)
                if not self.green_passed:
                    self.green_passed = True
                    self.pub_passed.publish(Bool(data=True))
                    rospy.loginfo("GREEN light detected, PASSED signal sent!")
            else:
                self.current_light = "NONE"
                # Keep previous stop state if no light detected
            
            # Publish state
            self.pub_light.publish(String(data=self.current_light))
            self.pub_stop.publish(Bool(data=self.stop_flag))
            
            if self.current_light != prev_light:
                rospy.loginfo(f"Traffic Light: {self.current_light}")
            
            # Draw visualization
            vis_img = img.copy()
            
            # Draw detected rectangles
            for rect in red_rects:
                cv2.rectangle(vis_img, (rect['x'], rect['y']), 
                             (rect['x'] + rect['w'], rect['y'] + rect['h']), (0, 0, 255), 3)
                cv2.putText(vis_img, "RED", (rect['x'], rect['y'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for rect in green_rects:
                cv2.rectangle(vis_img, (rect['x'], rect['y']), 
                             (rect['x'] + rect['w'], rect['y'] + rect['h']), (0, 255, 0), 3)
                cv2.putText(vis_img, "GREEN", (rect['x'], rect['y'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw state info
            color = (0, 0, 255) if self.current_light == "RED" else (0, 255, 0) if self.current_light == "GREEN" else (128, 128, 128)
            cv2.putText(vis_img, f"Light: {self.current_light}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(vis_img, f"Stop: {self.stop_flag}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Publish visualization
            self.publish_image_raw(self.pub_image, vis_img)
            self.publish_image_compressed(self.pub_image_compressed, vis_img)
            
            # Publish debug mask (combine red and green)
            debug_mask = cv2.merge([np.zeros_like(red_mask), green_mask, red_mask])
            self.publish_image_raw(self.pub_debug, debug_mask)
            self.publish_image_compressed(self.pub_debug_compressed, debug_mask)
            
        except Exception as e:
            rospy.logerr(f"Traffic light error: {e}")
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = TrafficLightNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
