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
        
        # Mission state
        self.current_state = ""
        
        # Red HSV thresholds (from yaml)
        red_h_low1 = rospy.get_param('~red_h_low1', 0)
        red_h_high1 = rospy.get_param('~red_h_high1', 10)
        red_h_low2 = rospy.get_param('~red_h_low2', 160)
        red_h_high2 = rospy.get_param('~red_h_high2', 180)
        red_s_low = rospy.get_param('~red_s_low', 100)
        red_s_high = rospy.get_param('~red_s_high', 255)
        red_v_low = rospy.get_param('~red_v_low', 100)
        red_v_high = rospy.get_param('~red_v_high', 255)
        
        self.RED_LOW1 = np.array([red_h_low1, red_s_low, red_v_low])
        self.RED_HIGH1 = np.array([red_h_high1, red_s_high, red_v_high])
        self.RED_LOW2 = np.array([red_h_low2, red_s_low, red_v_low])
        self.RED_HIGH2 = np.array([red_h_high2, red_s_high, red_v_high])
        
        # Green HSV thresholds (from yaml)
        green_h_low = rospy.get_param('~green_h_low', 35)
        green_h_high = rospy.get_param('~green_h_high', 85)
        green_s_low = rospy.get_param('~green_s_low', 23)
        green_s_high = rospy.get_param('~green_s_high', 255)
        green_v_low = rospy.get_param('~green_v_low', 80)
        green_v_high = rospy.get_param('~green_v_high', 255)
        
        self.GREEN_LOW = np.array([green_h_low, green_s_low, green_v_low])
        self.GREEN_HIGH = np.array([green_h_high, green_s_high, green_v_high])
        
        # LAB color space thresholds for Red
        self.RED_LAB_LOW = np.array([20, 130, 0])      # L, A, B
        self.RED_LAB_HIGH = np.array([255, 255, 255])
        
        # LAB color space thresholds for Green
        self.GREEN_LAB_LOW = np.array([40, 0, 0])      # L, A, B
        self.GREEN_LAB_HIGH = np.array([255, 127, 255])
        
        # RGB thresholds
        self.RGB_MIN_VALUE = 60  # Minimum intensity to avoid noise
        self.RGB_RATIO_THRESHOLD = 1.3  # Channel dominance ratio
        
        # Detection parameters (from yaml)
        self.min_area = rospy.get_param('~min_area', 300)
        self.circularity_thresh = rospy.get_param('~circularity_thresh', 0.6)
        
        # ROI settings (from yaml)
        self.roi_top = rospy.get_param('~roi_top', 200)
        self.roi_bottom = rospy.get_param('~roi_bottom', 350)
        self.roi_left = rospy.get_param('~roi_left', 0)
        self.roi_right = rospy.get_param('~roi_right', 250)
        
        # Brightness adjustment (from yaml)
        self.brightness_factor = rospy.get_param('~brightness_factor', 0.5)
        
        # Lane tracing parameters (from lane_detect_node)
        self.lane_speed = 0.25  # 기본값, lane_detect_node에서 수신
        
        # Lane steering from lane_detect_node
        self.lane_steering = 0.0
        
        # State
        self.current_light = "NONE"
        self.stop_flag = True  # 초기값 True: 초록불 감지 전까지 정지
        
        # Publishers - Image for rqt_image_view
        self.pub_image = rospy.Publisher('/limo/traffic_light/image', Image, queue_size=1)
        self.pub_debug = rospy.Publisher('/limo/traffic_light/debug', Image, queue_size=1)
        self.pub_traffic_state = rospy.Publisher('/limo/traffic_light/state', Bool, queue_size=1)
        # CompressedImage for bandwidth
        self.pub_image_compressed = rospy.Publisher('/limo/traffic_light/image/compressed', CompressedImage, queue_size=1)
        self.pub_debug_compressed = rospy.Publisher('/limo/traffic_light/debug/compressed', CompressedImage, queue_size=1)
        self.pub_light = rospy.Publisher('/limo/traffic_light', String, queue_size=1)
        self.pub_stop = rospy.Publisher('/limo/traffic_stop', Bool, queue_size=1)
        self.pub_passed = rospy.Publisher('/limo/traffic_passed', Bool, queue_size=1)
        
        # Lane state publisher (like final_pedestrian.py)
        self.pub_lane_state = rospy.Publisher('/state_manager/lane_state', String, queue_size=1)
        
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
        
        # Enable subscriber for sequence control
        self.sub_enable = rospy.Subscriber('/limo/traffic_enable', Bool, self.enable_callback, queue_size=1)
        
        # State subscriber from state_manager
        self.sub_state = rospy.Subscriber('/state_manager/state', String, self.state_callback, queue_size=1)
        
        # Dynamic reconfigure
        self.srv = Server(TrafficLightConfig, self.reconfigure_callback)
        
        # Control loop timer
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        
        rospy.loginfo("="*50)
        rospy.loginfo("Traffic Light Node initialized (HSV+LAB+RGB)")
        rospy.loginfo(f"lane_speed: {self.lane_speed} (from lane_detect_node)")
        rospy.loginfo("View: rqt_image_view /limo/traffic_light/image")
        rospy.loginfo("="*50)
    
    def state_callback(self, msg):
        """State callback from mission controller"""
        prev_state = self.current_state
        self.current_state = msg.data
        if prev_state != self.current_state:
            rospy.loginfo(f"Traffic light node received state: {self.current_state}")
            # Reset passed flag when entering WAITING_TRAFFIC_LIGHT state
            if self.current_state == "WAITING_TRAFFIC_LIGHT":
                self.green_passed = False
                self.stop_flag = True
    
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
        """Control loop - publish stop flag only (mission_controller handles cmd_vel)"""
        # Only publish stop flag when in WAITING_TRAFFIC_LIGHT state
        if self.current_state == "WAITING_TRAFFIC_LIGHT" or not self.current_state:
            self.pub_stop.publish(Bool(data=self.stop_flag))
        else:
            # 다른 상태에서는 항상 False 발행 (정지 해제)
            self.pub_stop.publish(Bool(data=False))
    
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
    
    def detect_color_hsv(self, hsv, low1, high1, low2=None, high2=None):
        """Detect color in HSV with optional second range (for red)"""
        mask1 = cv2.inRange(hsv, low1, high1)
        if low2 is not None and high2 is not None:
            mask2 = cv2.inRange(hsv, low2, high2)
            return cv2.bitwise_or(mask1, mask2)
        return mask1
    
    def detect_red_lab(self, lab):
        """Detect red color in LAB color space (A channel > 127)"""
        # LAB: L=[0,255], A=[0,255] (128=neutral, >128=red), B=[0,255]
        # Red: high A value (positive in A channel)
        mask = cv2.inRange(lab, self.RED_LAB_LOW, self.RED_LAB_HIGH)
        
        # Additional A channel filtering: A > 135 for strong red
        a_channel = lab[:, :, 1]
        a_mask = (a_channel > 135).astype(np.uint8) * 255
        
        mask = cv2.bitwise_and(mask, a_mask)
        return mask
    
    def detect_green_lab(self, lab):
        """Detect green color in LAB color space (A channel < 127)"""
        # Green: low A value (negative in A channel, <128 in OpenCV's 0-255 scale)
        mask = cv2.inRange(lab, self.GREEN_LAB_LOW, self.GREEN_LAB_HIGH)
        
        # Additional A channel filtering: A < 120 for strong green
        a_channel = lab[:, :, 1]
        a_mask = (a_channel < 120).astype(np.uint8) * 255
        
        mask = cv2.bitwise_and(mask, a_mask)
        return mask
    
    def detect_red_rgb(self, img):
        """Detect red color in RGB space (R dominant)"""
        b, g, r = cv2.split(img)
        
        # Red dominant: R > G and R > B with significant margin
        mask1 = ((r > g * self.RGB_RATIO_THRESHOLD) & (r > b * self.RGB_RATIO_THRESHOLD)).astype(np.uint8) * 255
        
        # Minimum intensity threshold to avoid dark noise
        mask2 = (r > self.RGB_MIN_VALUE).astype(np.uint8) * 255
        
        mask = cv2.bitwise_and(mask1, mask2)
        return mask
    
    def detect_green_rgb(self, img):
        """Detect green color in RGB space (G dominant)"""
        b, g, r = cv2.split(img)
        
        # Green dominant: G > R and G > B
        # For bright LED greens, use lower ratio threshold
        ratio_threshold = 1.1  # More lenient for bright green LEDs
        
        mask1 = ((g > r * ratio_threshold) & (g > b * ratio_threshold)).astype(np.uint8) * 255
        
        # Minimum intensity threshold - lower for bright LEDs
        mask2 = (g > 50).astype(np.uint8) * 255
        
        # Additional condition for very bright greens (phone screen)
        bright_green = ((g > 150) & (g > r) & (g > b)).astype(np.uint8) * 255
        
        mask = cv2.bitwise_or(cv2.bitwise_and(mask1, mask2), bright_green)
        return mask
    
    def detect_color_multi_space(self, img, hsv, lab, color_type):
        """Detect color using HSV + LAB + RGB color spaces combined"""
        if color_type == "RED":
            # HSV detection
            hsv_mask = self.detect_color_hsv(hsv, self.RED_LOW1, self.RED_HIGH1, 
                                            self.RED_LOW2, self.RED_HIGH2)
            # LAB detection
            lab_mask = self.detect_red_lab(lab)
            
            # RGB detection
            rgb_mask = self.detect_red_rgb(img)
            
            # Combine masks with OR operation
            combined_mask = cv2.bitwise_or(hsv_mask, lab_mask)
            combined_mask = cv2.bitwise_or(combined_mask, rgb_mask)
            
        elif color_type == "GREEN":
            # HSV detection
            hsv_mask = self.detect_color_hsv(hsv, self.GREEN_LOW, self.GREEN_HIGH)
            
            # LAB detection
            lab_mask = self.detect_green_lab(lab)
            
            # RGB detection
            rgb_mask = self.detect_green_rgb(img)
            
            # Combine masks with OR operation
            combined_mask = cv2.bitwise_or(hsv_mask, lab_mask)
            combined_mask = cv2.bitwise_or(combined_mask, rgb_mask)
        else:
            combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        return combined_mask
    
    def find_circular_contours(self, mask, color_name):
        """Find circular contours from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 원형 판별
            if circularity >= self.circularity_thresh:
                (x, y), radius = cv2.minEnclosingCircle(cnt)

                circles.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'area': area,
                    'circularity': circularity,
                    'color': color_name
                })

        return circles
    
    def image_callback(self, msg):
        """Image callback"""
        # Skip processing if disabled
        if not self.enabled:
            return
        
        # Only process when in WAITING_TRAFFIC_LIGHT state or no state set yet
        if self.current_state and self.current_state != "WAITING_TRAFFIC_LIGHT":
            return
            
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            full_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if full_img is None:
                return
            
            # Extract ROI
            h, w = full_img.shape[:2]
            roi_top = max(0, self.roi_top)
            roi_bottom = min(h, self.roi_bottom)
            roi_left = max(0, self.roi_left)
            roi_right = min(w, self.roi_right)
            
            roi_original = full_img[roi_top:roi_bottom, roi_left:roi_right].copy()
            
            # Adaptive brightness adjustment
            hsv0 = cv2.cvtColor(roi_original, cv2.COLOR_BGR2HSV)
            v_mean = hsv0[:, :, 2].mean()

            alpha = 1.0
            if v_mean > 120:
                alpha = self.brightness_factor

            img = cv2.convertScaleAbs(roi_original, alpha=alpha, beta=0)

            # Convert to multiple color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Multi-space color detection
            red_mask = self.detect_color_multi_space(img, hsv, lab, "RED")
            green_mask = self.detect_color_multi_space(img, hsv, lab, "GREEN")
            
            # Apply morphological operations to clean masks
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

            # Find circular contours
            red_circles = self.find_circular_contours(red_mask, "RED")
            green_circles = self.find_circular_contours(green_mask, "GREEN")
            
            # WAITING_TRAFFIC_LIGHT 상태일 때는 항상 LANE_FOLLOWING 발행
            self.pub_lane_state.publish(String(data="LANE_FOLLOWING"))
            
            # Determine traffic light state
            prev_light = self.current_light
            
            if red_circles:
                self.current_light = "RED"
                self.stop_flag = True
                self.pub_traffic_state.publish(Bool(data=True))
            elif green_circles:
                self.current_light = "GREEN"
                self.stop_flag = False
                self.pub_traffic_state.publish(Bool(data=False))

                # Signal green light passed (only once)
                if not self.green_passed:
                    self.green_passed = True
                    self.pub_passed.publish(Bool(data=True))
                    rospy.loginfo("GREEN light detected, PASSED signal sent!")
            else:
                self.current_light = "NONE"
            
            # Publish state
            self.pub_light.publish(String(data=self.current_light))
            
            if self.current_light != prev_light:
                rospy.loginfo(f"Traffic Light: {self.current_light}")
            
            # Draw visualization
            vis_img = img.copy()
            
            # Draw detected circles
            for c in red_circles:
                cv2.circle(vis_img, c['center'], c['radius'], (0, 0, 255), 3)
                cv2.putText(vis_img, "RED", c['center'],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for c in green_circles:
                cv2.circle(vis_img, c['center'], c['radius'], (0, 255, 0), 3)
                cv2.putText(vis_img, "GREEN", c['center'],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw state info
            color = (0, 0, 255) if self.current_light == "RED" else (0, 255, 0) if self.current_light == "GREEN" else (128, 128, 128)
            cv2.putText(vis_img, f"Light: {self.current_light}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(vis_img, f"Stop: {self.stop_flag}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis_img, "HSV+LAB+RGB", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
