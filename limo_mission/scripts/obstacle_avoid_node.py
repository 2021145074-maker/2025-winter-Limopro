#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obstacle Avoidance Node for LIMO Pro (Gap Finding Algorithm)
- Uses LiDAR to scan front ±30 degrees
- Divides into sectors and finds the best gap
- Steers toward the widest open gap
- Returns to lane tracing when path is clear
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from limo_mission.cfg import ObstacleAvoidConfig


class ObstacleAvoidNode:
    def __init__(self):
        rospy.init_node('obstacle_avoid_node')
        
        # Parameters (from yaml/cfg)
        self.scan_angle = rospy.get_param('~scan_angle', 30.0)
        self.safe_distance = rospy.get_param('~safe_distance', 0.5)
        self.stop_distance = rospy.get_param('~stop_distance', 0.2)
        self.num_sectors = rospy.get_param('~num_sectors', 12)
        
        # Lane speed from lane_detect_node
        self.lane_speed = 0.25  # 기본값, lane_detect_node에서 수신
        self.avoid_speed = rospy.get_param('~avoid_speed', 0.2)  # 회피 전용 속도
        self.max_steering = rospy.get_param('~max_steering', 0.5)
        self.steering_gain = rospy.get_param('~steering_gain', 0.02)
        
        # State
        self.avoiding = False
        self.clear_count = 0
        self.clear_threshold = rospy.get_param('~clear_threshold', 20)
        self.last_state = ""
        
        # Sector data
        self.sector_distances = [10.0] * self.num_sectors
        self.sector_angles = []
        
        # Lane steering from lane_detect_node
        self.lane_steering = 0.0
        
        # LiDAR data
        self.ranges = None
        self.angle_increment = 0
        
        # Publishers
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_state = rospy.Publisher('/limo/obstacle_state', String, queue_size=1)
        self.pub_debug_image = rospy.Publisher('/limo/obstacle/debug', Image, queue_size=1)
        
        # CV Bridge
        self.bridge = CvBridge()
        self.vis_size = 400
        
        # Subscribers
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.sub_steering = rospy.Subscriber('/limo/steering_offset', Float32, self.steering_callback, queue_size=1)
        self.sub_speed = rospy.Subscriber('/limo/lane_speed', Float32, self.speed_callback, queue_size=1)
        
        # Dynamic reconfigure
        self.srv = Server(ObstacleAvoidConfig, self.reconfigure_callback)
        
        # Timers
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        self.vis_timer = rospy.Timer(rospy.Duration(0.1), self.visualization_callback)
        
        self._init_sectors()
        
        rospy.loginfo("="*50)
        rospy.loginfo("Obstacle Avoid Node initialized (Gap Finding)")
        rospy.loginfo(f"Scan: ±{self.scan_angle}°, Safe: {self.safe_distance}m")
        rospy.loginfo("View: rqt_image_view /limo/obstacle/debug")
        rospy.loginfo("="*50)
    
    def _init_sectors(self):
        """Initialize sector center angles"""
        total_angle = self.scan_angle * 2
        sector_size = total_angle / self.num_sectors
        
        self.sector_angles = []
        for i in range(self.num_sectors):
            angle = self.scan_angle - sector_size * (i + 0.5)
            self.sector_angles.append(angle)
        
        self.sector_distances = [10.0] * self.num_sectors
    
    def steering_callback(self, msg):
        self.lane_steering = msg.data
    
    def speed_callback(self, msg):
        """Speed callback from lane_detect_node"""
        self.lane_speed = msg.data
    
    def reconfigure_callback(self, config, level):
        self.scan_angle = config.get('scan_angle', self.scan_angle)
        self.safe_distance = config.get('safe_distance', self.safe_distance)
        self.stop_distance = config.get('stop_distance', self.stop_distance)
        self.num_sectors = config.get('num_sectors', self.num_sectors)
        # lane_speed는 lane_detect_node에서 관리, avoid_speed만 자체 관리
        self.avoid_speed = config.get('avoid_speed', self.avoid_speed)
        self.max_steering = config.get('max_steering', self.max_steering)
        self.steering_gain = config.get('steering_gain', self.steering_gain)
        self.clear_threshold = config.get('clear_threshold', self.clear_threshold)
        self._init_sectors()
        rospy.loginfo(f"Updated: scan=±{self.scan_angle}°, safe={self.safe_distance}m, clear={self.clear_threshold}")
        return config
    
    def scan_callback(self, msg):
        self.ranges = np.array(msg.ranges)
        self.angle_increment = msg.angle_increment
        self._calculate_sector_distances()
    
    def _calculate_sector_distances(self):
        if self.ranges is None or self.angle_increment == 0:
            return
        
        total_points = len(self.ranges)
        center_idx = total_points // 2
        points_per_degree = 1.0 / np.degrees(self.angle_increment)
        sector_size_deg = (self.scan_angle * 2) / self.num_sectors
        
        for i in range(self.num_sectors):
            start_angle = -self.scan_angle + sector_size_deg * i
            end_angle = start_angle + sector_size_deg
            
            start_idx = int(center_idx + start_angle * points_per_degree)
            end_idx = int(center_idx + end_angle * points_per_degree)
            
            start_idx = max(0, min(start_idx, total_points - 1))
            end_idx = max(0, min(end_idx, total_points - 1))
            
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            sector_ranges = self.ranges[start_idx:end_idx + 1]
            valid = sector_ranges[(sector_ranges > 0.01) & (sector_ranges < 10.0)]
            
            self.sector_distances[i] = np.min(valid) if len(valid) > 0 else 10.0
    
    def _find_best_gap(self):
        """Find the best gap (widest open space)"""
        is_open = [d >= self.safe_distance for d in self.sector_distances]
        
        gaps = []
        gap_start = None
        
        for i, open_sector in enumerate(is_open):
            if open_sector and gap_start is None:
                gap_start = i
            elif not open_sector and gap_start is not None:
                gaps.append((gap_start, i - 1))
                gap_start = None
        
        if gap_start is not None:
            gaps.append((gap_start, len(is_open) - 1))
        
        if not gaps:
            max_idx = np.argmax(self.sector_distances)
            return self.sector_angles[max_idx], 1
        
        best_gap = max(gaps, key=lambda g: g[1] - g[0])
        gap_start_idx, gap_end_idx = best_gap
        gap_width = gap_end_idx - gap_start_idx + 1
        
        center_idx = (gap_start_idx + gap_end_idx) / 2.0
        if center_idx == int(center_idx):
            gap_center_angle = self.sector_angles[int(center_idx)]
        else:
            low_idx = int(center_idx)
            high_idx = min(low_idx + 1, len(self.sector_angles) - 1)
            ratio = center_idx - low_idx
            gap_center_angle = (self.sector_angles[low_idx] * (1 - ratio) + 
                               self.sector_angles[high_idx] * ratio)
        
        return gap_center_angle, gap_width
    
    def _has_obstacle_in_front(self):
        center_sectors = self.num_sectors // 3
        start = (self.num_sectors - center_sectors) // 2
        end = start + center_sectors
        front_distances = self.sector_distances[start:end]
        return min(front_distances) < self.safe_distance if front_distances else False
    
    def _get_min_front_distance(self):
        center_sectors = self.num_sectors // 3
        start = (self.num_sectors - center_sectors) // 2
        end = start + center_sectors
        front_distances = self.sector_distances[start:end]
        return min(front_distances) if front_distances else 10.0
    
    def publish_debug_image(self):
        """LiDAR sector visualization"""
        size = self.vis_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        cx, cy = size // 2, size - 50
        
        # Grid
        for r_meters in [0.5, 1.0, 1.5, 2.0]:
            r_pixels = int(r_meters * 100)
            cv2.circle(img, (cx, cy), r_pixels, (40, 40, 40), 1)
        
        # Sectors
        for i, (angle, dist) in enumerate(zip(self.sector_angles, self.sector_distances)):
            rad = np.radians(90 - angle)
            display_dist = min(dist, 2.0)
            r = int(display_dist * 100)
            
            ex = int(cx + r * np.cos(rad))
            ey = int(cy - r * np.sin(rad))
            
            if dist >= self.safe_distance:
                color = (0, 255, 0)
            elif dist >= self.stop_distance:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.line(img, (cx, cy), (ex, ey), color, 2)
            if dist < 2.0:
                cv2.circle(img, (ex, ey), 4, color, -1)
        
        # Best gap arrow
        gap_angle, gap_width = self._find_best_gap()
        gap_rad = np.radians(90 - gap_angle)
        gap_len = 150
        gap_x = int(cx + gap_len * np.cos(gap_rad))
        gap_y = int(cy - gap_len * np.sin(gap_rad))
        cv2.arrowedLine(img, (cx, cy), (gap_x, gap_y), (255, 255, 0), 3, tipLength=0.2)
        
        # Robot
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
        
        # Info panel
        cv2.rectangle(img, (0, 0), (size, 70), (0, 0, 0), -1)
        
        state_text = self.last_state if self.last_state else "IDLE"
        if "AVOIDING" in state_text:
            state_color = (0, 255, 255)
        elif "LANE" in state_text:
            state_color = (0, 255, 0)
        else:
            state_color = (128, 128, 128)
        
        cv2.putText(img, "OBSTACLE AVOID", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"State: {state_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        cv2.putText(img, f"Gap: {gap_angle:.1f}deg", (size - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        try:
            self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(5, f"Debug image error: {e}")
    
    def visualization_callback(self, event):
        if self.ranges is not None:
            self.publish_debug_image()
    
    def control_loop(self, event):
        cmd = Twist()
        state = "IDLE"
        
        # Too close - stop
        if self._get_min_front_distance() < self.stop_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            state = "TOO_CLOSE"
            self.avoiding = True
            self.clear_count = 0
        
        # Obstacle in front - avoid
        elif self._has_obstacle_in_front():
            self.avoiding = True
            self.clear_count = 0
            
            gap_angle, gap_width = self._find_best_gap()
            steering = -self.steering_gain * gap_angle
            steering = np.clip(steering, -self.max_steering, self.max_steering)
            
            cmd.linear.x = self.avoid_speed
            cmd.angular.z = steering
            state = f"AVOIDING (gap={gap_angle:.1f}°)"
        
        # Avoiding mode - hysteresis
        elif self.avoiding:
            self.clear_count += 1
            
            if self.clear_count >= self.clear_threshold:
                self.avoiding = False
                self.clear_count = 0
                cmd.linear.x = self.lane_speed
                cmd.angular.z = -self.lane_steering
                state = "LANE_TRACING"
            else:
                gap_angle, _ = self._find_best_gap()
                steering = -self.steering_gain * gap_angle
                steering = np.clip(steering, -self.max_steering, self.max_steering)
                cmd.linear.x = self.avoid_speed
                cmd.angular.z = steering
                state = f"AVOIDING_CLEAR ({self.clear_count}/{self.clear_threshold})"
        
        # Clear - lane tracing (lane_detect_node의 속도 사용)
        else:
            cmd.linear.x = self.lane_speed
            cmd.angular.z = -self.lane_steering
            state = "LANE_TRACING"
        
        if state != self.last_state:
            rospy.loginfo(f"[OBSTACLE] {state}")
            self.last_state = state
        
        self.pub_cmd.publish(cmd)
        self.pub_state.publish(String(data=state))
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ObstacleAvoidNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
