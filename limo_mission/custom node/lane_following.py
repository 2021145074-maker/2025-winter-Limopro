#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Lane Detection Node with Pure Pursuit Control
- Bird's Eye View transformation
- Histogram-based sliding window lane detection
- Single/Dual lane handling with offset
- Pure Pursuit with PD control
- State manager integration
"""



import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time



class LaneDetectionNode:
    def __init__(self):
        rospy.init_node('lane_following_node')
        self.bridge = CvBridge()
        
        # White lane detection parameters
        self.WHITE_LOWER = np.array([0, 200, 0])
        self.WHITE_UPPER = np.array([255, 255, 200])
        
        # Bird's Eye View parameters
        self.img_width = 640
        self.img_height = 480
        self.roi_height = 240
        
        # BEV near-range keep
        self.bev_keep_bottom = 320
        
        # Source points (trapezoid in original image)
        self.src_points = np.float32([
            [80.0, 310.0],
            [560.0, 310.0],
            [640, 480],
            [0, 480]
        ])
        
        # Destination points (rectangle in BEV)
        self.dst_points = np.float32([
            [120, 0],
            [520, 0],
            [520, 480],
            [120, 480]
        ])
        
        # Compute perspective transform matrices
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        
        # Sliding window parameters
        self.nwindows = 9
        self.margin = 50
        self.minpix = 50
        
        # Lane tracking
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.prev_left_x = None
        self.prev_right_x = None
        self.track_fail_count = 0
        self.max_track_fail = 30
        
        # Lane offset
        self.lane_width_m = 0.6
        self.lane_offset = None
        
        # Pure Pursuit parameters
        self.wheelbase = 0.2
        
        # === 속도 적응형 파라미터 설정 ===
        self.speed_low = 0.4
        self.speed_high = 0.8
        
        # Look-ahead distance
        self.look_ahead_low = 100
        self.look_ahead_high = 70  # 기본값
        
        # ✅ State별 look-ahead 설정
        self.look_ahead_high_default = 70          # 일반 모드
        self.look_ahead_high_traffic = 100         # 🚦 WAITING_TRAFFIC_LIGHT 모드
        self.look_ahead_high_obstacle = 120        # OBSTACLE_AVOIDANCE 모드
        
        self.look_ahead_distance = 100
        
        # ✅ PD gains - 일반 모드 (Default)
        self.kp_lateral_low_default = 0.014
        self.kp_lateral_high_default = 0.016
        self.kd_lateral_low_default = 0.001
        self.kd_lateral_high_default = 0.001
        
        self.kp_heading_low_default = 0.65
        self.kp_heading_high_default = 0.85
        self.kd_heading_low_default = 0.05
        self.kd_heading_high_default = 0.04
        
        # ✅ PD gains - OBSTACLE_AVOIDANCE 모드
        self.kp_lateral_low_obstacle = 0.012
        self.kp_lateral_high_obstacle = 0.014
        self.kd_lateral_low_obstacle = 0.001
        self.kd_lateral_high_obstacle = 0.001
        
        self.kp_heading_low_obstacle = 0.6
        self.kp_heading_high_obstacle = 0.80
        self.kd_heading_low_obstacle = 0.05
        self.kd_heading_high_obstacle = 0.04
        
        # ✅ 현재 사용 중인 PD gain 범위 (동적 변경)
        self.kp_lateral_low = self.kp_lateral_low_default
        self.kp_lateral_high = self.kp_lateral_high_default
        self.kd_lateral_low = self.kd_lateral_low_default
        self.kd_lateral_high = self.kd_lateral_high_default
        
        self.kp_heading_low = self.kp_heading_low_default
        self.kp_heading_high = self.kp_heading_high_default
        self.kd_heading_low = self.kd_heading_low_default
        self.kd_heading_high = self.kd_heading_high_default
        
        # 현재 PD 값
        self.kp_lateral = 0.008
        self.kd_lateral = 0.001
        self.kp_heading = 0.5
        self.kd_heading = 0.05
        self.prev_lateral_error = 0.0
        self.prev_heading_error = 0.0
        
        # Speed control
        self.base_speed = 0.8
        self.default_base_speed = 0.8
        self.max_angular_vel = 1.2
        
        # State management
        self.current_state = None
        self.mission_state = None  # ✅ 추가: 미션 상태 추적
        self.enabled = False
        self.traffic_stop = False
        
        # 장애물 회피 시 단일 차선 모드
        self.avoid_side = "NONE"
        
        # 차선 하나만 보일 때 파라미터들
        bev_lane_width_pixel = abs(self.dst_points[1][0] - self.dst_points[0][0])
        self.ym_per_pix = 0.01
        self.xm_per_pix = self.lane_width_m / bev_lane_width_pixel
        
        self.lane_offset = int(bev_lane_width_pixel / 2.0)
        self.avoid_lane_offset = int(bev_lane_width_pixel * 0.7)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage,
                                         self.image_callback, queue_size=1, buff_size=2**24)
        self.state_sub = rospy.Subscriber('/state_manager/lane_state', String, self.state_callback, queue_size=1)
        self.traffic_stop_sub = rospy.Subscriber('/limo/traffic_stop', Bool, self.traffic_stop_callback, queue_size=1)
        self.base_speed_sub = rospy.Subscriber('/lane_detection/base_speed', Float32, self.base_speed_callback, queue_size=1)
        self.avoid_side_sub = rospy.Subscriber('/limo/obstacle/avoid_side', String, self.avoid_side_callback, queue_size=1)
        
        # ✅ 추가: 미션 상태 구독
        self.mission_state_sub = rospy.Subscriber('/state_manager/state', String, self.mission_state_callback, queue_size=1)
        
        # Publishers
        self.white_image_pub = rospy.Publisher('/lane_detection/white_lanes/compressed', CompressedImage, queue_size=1)
        self.edge_image_pub = rospy.Publisher('/lane_detection/mask/compressed', CompressedImage, queue_size=1)
        self.bev_image_pub = rospy.Publisher('/lane_detection/bev/compressed', CompressedImage, queue_size=1)
        self.lane_image_pub = rospy.Publisher('/lane_detection/image/compressed', CompressedImage, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Control hold/timeout
        self.last_good_time = rospy.Time.now().to_sec()
        self.last_speed = 0.0
        self.last_angular = 0.0
        self.HOLD_SEC = 0.7
        self.STOP_SEC = 2.00
        
        # 초기 속도에 맞게 적응형 파라미터 설정
        self.update_speed_adaptive_params()
        
        rospy.loginfo("="*50)
        rospy.loginfo("Advanced Lane Detection Node Initialized")
        rospy.loginfo(f"Base speed: {self.base_speed:.2f} m/s")
        rospy.loginfo(f"Look-ahead: {self.look_ahead_distance} px")
        rospy.loginfo("Waiting for state: LANE_FOLLOWING")
        rospy.loginfo("="*50)



    def mission_state_callback(self, msg):
        """미션 상태 콜백 - State별 look-ahead와 PD gains 설정"""
        prev_mission = self.mission_state
        self.mission_state = msg.data
        
        if prev_mission != self.mission_state:
            rospy.loginfo(f"[Lane] Mission state changed: {prev_mission} -> {self.mission_state}")
            
            # ✅ State 변경 시 look-ahead 재설정
            if self.mission_state == "OBSTACLE_AVOIDANCE":
                self.look_ahead_high = self.look_ahead_high_obstacle  # 120
                
                # ✅ OBSTACLE_AVOIDANCE용 PD gains로 전환
                self.kp_lateral_low = self.kp_lateral_low_obstacle
                self.kp_lateral_high = self.kp_lateral_high_obstacle
                self.kd_lateral_low = self.kd_lateral_low_obstacle
                self.kd_lateral_high = self.kd_lateral_high_obstacle
                
                self.kp_heading_low = self.kp_heading_low_obstacle
                self.kp_heading_high = self.kp_heading_high_obstacle
                self.kd_heading_low = self.kd_heading_low_obstacle
                self.kd_heading_high = self.kd_heading_high_obstacle
                
                rospy.loginfo(f"[Lane] OBSTACLE_AVOIDANCE mode - look_ahead_high={self.look_ahead_high}, "
                             f"kp_lat=[{self.kp_lateral_low:.3f}-{self.kp_lateral_high:.3f}], "
                             f"kp_head=[{self.kp_heading_low:.2f}-{self.kp_heading_high:.2f}]")
            
            elif self.mission_state == "WAITING_TRAFFIC_LIGHT":
                self.look_ahead_high = self.look_ahead_high_traffic  # 100
                
                # ✅ 일반 PD gains로 복귀
                self.kp_lateral_low = self.kp_lateral_low_default
                self.kp_lateral_high = self.kp_lateral_high_default
                self.kd_lateral_low = self.kd_lateral_low_default
                self.kd_lateral_high = self.kd_lateral_high_default
                
                self.kp_heading_low = self.kp_heading_low_default
                self.kp_heading_high = self.kp_heading_high_default
                self.kd_heading_low = self.kd_heading_low_default
                self.kd_heading_high = self.kd_heading_high_default
                
                rospy.loginfo(f"[Lane] WAITING_TRAFFIC_LIGHT mode - look_ahead_high={self.look_ahead_high}")
            
            else:
                self.look_ahead_high = self.look_ahead_high_default  # 70
                
                # ✅ 일반 PD gains로 복귀
                self.kp_lateral_low = self.kp_lateral_low_default
                self.kp_lateral_high = self.kp_lateral_high_default
                self.kd_lateral_low = self.kd_lateral_low_default
                self.kd_lateral_high = self.kd_lateral_high_default
                
                self.kp_heading_low = self.kp_heading_low_default
                self.kp_heading_high = self.kp_heading_high_default
                self.kd_heading_low = self.kd_heading_low_default
                self.kd_heading_high = self.kd_heading_high_default
                
                rospy.loginfo(f"[Lane] Normal mode - look_ahead_high={self.look_ahead_high}")
            
            # look-ahead와 PD 파라미터 즉시 업데이트
            self.update_speed_adaptive_params()



    def state_callback(self, msg):
        """State callback from lane_state topic"""
        self.current_state = msg.data
        self.enabled = (self.current_state.upper() == "LANE_FOLLOWING")
        
        if self.current_state.upper() == "WAITING_PEDESTRIAN":
            self.publish_cmd(0.0, 0.0, mark_good=False)
            rospy.loginfo_throttle(0.5, "Lane: WAITING_PEDESTRIAN - STOP")
        
        rospy.loginfo_throttle(1.0, f"Lane State: {self.current_state}, Enabled: {self.enabled}")



    def traffic_stop_callback(self, msg):
        """Traffic stop callback"""
        self.traffic_stop = msg.data
        if self.traffic_stop:
            rospy.loginfo_throttle(1.0, "Traffic STOP signal received")



    def base_speed_callback(self, msg):
        """Base speed callback"""
        new_speed = msg.data
        if 0.0 <= new_speed <= 1.0:
            self.base_speed = new_speed
            self.update_speed_adaptive_params()
            rospy.loginfo_throttle(1.0, f"Base speed changed to: {self.base_speed:.2f}")



    def avoid_side_callback(self, msg):
        """장애물 회피 방향 콜백"""
        prev_side = self.avoid_side
        self.avoid_side = msg.data.upper()
        
        if prev_side != self.avoid_side:
            rospy.loginfo(f"[Lane] Avoid side changed: {prev_side} -> {self.avoid_side}")
            
            # 모드 변경 시 이전 차선 정보 리셋하여 새로 탐지
            self.prev_left_fit = None
            self.prev_right_fit = None
            self.prev_left_x = None
            self.prev_right_x = None
            self.track_fail_count = 0



    def update_speed_adaptive_params(self):
        """속도에 따라 PD gains와 look-ahead distance를 동적으로 조정"""
        speed = self.base_speed
        
        # 속도를 0~1 범위로 정규화
        t = np.clip((speed - self.speed_low) / (self.speed_high - self.speed_low + 1e-6), 0.0, 1.0)
        
        # ✅ look-ahead_high는 이미 mission_state_callback에서 설정됨
        # 선형 보간으로 현재 look-ahead 계산
        self.look_ahead_distance = int(self.look_ahead_low + t * (self.look_ahead_high - self.look_ahead_low))
        
        # ✅ Lateral PD gains (현재 state에 맞는 범위 사용)
        self.kp_lateral = self.kp_lateral_low + t * (self.kp_lateral_high - self.kp_lateral_low)
        self.kd_lateral = self.kd_lateral_low + t * (self.kd_lateral_high - self.kd_lateral_low)
        
        # ✅ Heading PD gains (현재 state에 맞는 범위 사용)
        self.kp_heading = self.kp_heading_low + t * (self.kp_heading_high - self.kp_heading_low)
        self.kd_heading = self.kd_heading_low + t * (self.kd_heading_high - self.kd_heading_low)
        
        rospy.loginfo_throttle(2.0,
            f"[Adaptive] state={self.mission_state} speed={speed:.2f} | LA={self.look_ahead_distance} | "
            f"kp_lat={self.kp_lateral:.4f} kd_lat={self.kd_lateral:.4f} | "
            f"kp_head={self.kp_heading:.3f} kd_head={self.kd_heading:.3f}")



    def publish_cmd(self, speed, angular, mark_good=True):
        """Publish Twist cmd"""
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(angular)
        self.cmd_vel_pub.publish(cmd)
        
        if mark_good:
            self.last_good_time = rospy.Time.now().to_sec()
            self.last_speed = float(speed)
            self.last_angular = float(angular)



    def image_callback(self, msg):
        """Main image processing callback"""
        if not self.enabled:
            if self.current_state and self.current_state.upper() == "WAITING_PEDESTRIAN":
                self.publish_cmd(0.0, 0.0, mark_good=False)
            return
        
        if self.traffic_stop:
            self.publish_cmd(0.0, 0.0, mark_good=False)
            return
        
        try:
            # Decode image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
            # 1) Build binary lane mask
            lane_binary = self.detect_lane_binary(cv_image)
            
            # 2) Warp to BEV
            bev_binary = cv2.warpPerspective(lane_binary, self.M, (self.img_width, self.img_height))
            
            # Keep only near-range
            if self.bev_keep_bottom is not None and 0 < self.bev_keep_bottom < self.img_height:
                cut = self.img_height - int(self.bev_keep_bottom)
                if cut > 0:
                    bev_binary[:cut, :] = 0
            
            # 3) Lane detection
            left_fit = right_fit = None
            out_img = None
            ok = False
            
            # Dynamic tracking margin
            dyn_margin = max(self.margin, 80 + 30 * min(self.track_fail_count, 4))
            if abs(getattr(self, 'last_angular', 0.0)) > 0.6:
                dyn_margin = max(dyn_margin, 160)
            
            use_tracking = ((self.prev_left_fit is not None) or (self.prev_right_fit is not None)) and (self.track_fail_count < self.max_track_fail)
            
            if use_tracking:
                left_fit, right_fit, out_img, ok = self.detect_lanes_from_prior(
                    bev_binary, self.prev_left_fit, self.prev_right_fit, margin=dyn_margin
                )
            
            if (not ok) or (out_img is None):
                left_fit, right_fit, out_img = self.detect_lanes_sliding_window(bev_binary)
                ok = (left_fit is not None) or (right_fit is not None)
            
            # Update tracking state
            if ok:
                self.prev_left_fit = left_fit if left_fit is not None else self.prev_left_fit
                self.prev_right_fit = right_fit if right_fit is not None else self.prev_right_fit
                self.track_fail_count = 0
                
                y_eval = bev_binary.shape[0] - 1
                if left_fit is not None:
                    self.prev_left_x = int(left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2])
                if right_fit is not None:
                    self.prev_right_x = int(right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2])
            else:
                self.track_fail_count += 1
            
            # 4) Generate driving path
            driving_path = self.generate_driving_path(left_fit, right_fit, bev_binary.shape)
            
            # 5) Control vehicle
            if driving_path is not None:
                self.pure_pursuit_control(driving_path, out_img)
            else:
                now = rospy.Time.now().to_sec()
                dt = now - self.last_good_time
                
                if dt < self.HOLD_SEC:
                    self.publish_cmd(self.last_speed, self.last_angular, mark_good=False)
                elif dt < self.STOP_SEC:
                    k = (self.STOP_SEC - dt) / (self.STOP_SEC - self.HOLD_SEC)
                    v = max(0.0, self.last_speed * k)
                    self.publish_cmd(v, self.last_angular, mark_good=False)
                else:
                    self.publish_cmd(0.15, 0.0, mark_good=False)
                    rospy.logwarn_throttle(1.0, "[lane_detection] Lane lost - slow straight")
            
            # 6) Publish BEV visualization
            if out_img is None:
                out_img = cv2.cvtColor((bev_binary > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            
            bev_msg = self.bridge.cv2_to_compressed_imgmsg(out_img)
            self.bev_image_pub.publish(bev_msg)
        
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")



    def detect_white_lanes(self, image):
        """Detect white lanes and return binary mask"""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hls = cv2.GaussianBlur(hls, (5, 5), 0)
        white_mask = cv2.inRange(hls, self.WHITE_LOWER, self.WHITE_UPPER)
        
        roi_mask = np.zeros_like(white_mask)
        roi_mask[self.roi_height:, :] = white_mask[self.roi_height:, :]
        
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        white_vis = cv2.bitwise_and(image, image, mask=roi_mask)
        rosImage_white = self.bridge.cv2_to_compressed_imgmsg(white_vis)
        self.white_image_pub.publish(rosImage_white)
        
        return roi_mask



    def detect_lane_binary(self, bgr_image):
        """Build a binary lane mask"""
        white_mask = self.detect_white_lanes(bgr_image)
        
        mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        try:
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.edge_image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(vis))
        except Exception:
            pass
        
        return mask



    def detect_lanes_from_prior(self, binary_warped, left_fit_prev, right_fit_prev, margin=80):
        """Fast lane tracking around previous polynomials"""
        if binary_warped is None or binary_warped.size == 0:
            return None, None, None, False
        
        bw = (binary_warped > 0).astype(np.uint8) * 255
        nonzero = bw.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        out_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        
        left_fit = None
        right_fit = None
        left_lane_inds = np.array([], dtype=np.int64)
        right_lane_inds = np.array([], dtype=np.int64)
        
        if left_fit_prev is not None:
            left_x_pred = left_fit_prev[0] * (nonzeroy**2) + left_fit_prev[1] * nonzeroy + left_fit_prev[2]
            left_lane_inds = np.where((nonzerox > (left_x_pred - margin)) & (nonzerox < (left_x_pred + margin)))[0]
        
        if right_fit_prev is not None:
            right_x_pred = right_fit_prev[0] * (nonzeroy**2) + right_fit_prev[1] * nonzeroy + right_fit_prev[2]
            right_lane_inds = np.where((nonzerox > (right_x_pred - margin)) & (nonzerox < (right_x_pred + margin)))[0]
        
        if left_lane_inds.size > 150:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            left_fit = np.polyfit(lefty, leftx, 2)
            out_img[lefty, leftx] = (255, 0, 0)
        
        if right_lane_inds.size > 150:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            right_fit = np.polyfit(righty, rightx, 2)
            out_img[righty, rightx] = (0, 0, 255)
        
        ok = (left_fit is not None) or (right_fit is not None)
        
        if left_fit is not None and right_fit is not None:
            y_eval = bw.shape[0] - 1
            left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            lane_w = right_x - left_x
            
            if not (140 <= lane_w <= 520):
                ok = False
        
        return left_fit, right_fit, out_img, ok



    def find_histogram_peaks(self, histogram, threshold, min_distance=80):
        """히스토그램에서 유의미한 피크들 찾기"""
        peaks = []
        h = histogram.copy()
        
        while True:
            max_idx = np.argmax(h)
            max_val = h[max_idx]
            
            if max_val < threshold:
                break
            
            peaks.append((max_idx, max_val))
            
            left_bound = max(0, max_idx - min_distance)
            right_bound = min(len(h), max_idx + min_distance)
            h[left_bound:right_bound] = 0
            
            if len(peaks) >= 3:
                break
        
        return peaks



    def assign_lane_peaks(self, peaks, histogram, vehicle_center):
        """피크들을 좌/우 차선에 할당"""
        leftx_base = None
        rightx_base = None
        left_detected = False
        right_detected = False
        
        if len(peaks) == 0:
            return leftx_base, rightx_base, left_detected, right_detected
        
        peak_xs = [p[0] for p in peaks]
        
        if len(peaks) >= 2:
            peak_xs_sorted = sorted(peak_xs)
            leftx_base = peak_xs_sorted[0]
            rightx_base = peak_xs_sorted[-1]
            
            if abs(rightx_base - leftx_base) < 120:
                single_x = (leftx_base + rightx_base) // 2
                
                if self.prev_left_x is not None and self.prev_right_x is not None:
                    dist_to_left = abs(single_x - self.prev_left_x)
                    dist_to_right = abs(single_x - self.prev_right_x)
                    
                    if dist_to_left < dist_to_right:
                        leftx_base = single_x
                        rightx_base = None
                    else:
                        rightx_base = single_x
                        leftx_base = None
                elif single_x < vehicle_center:
                    leftx_base = single_x
                    rightx_base = None
                else:
                    rightx_base = single_x
                    leftx_base = None
            
            left_detected = leftx_base is not None
            right_detected = rightx_base is not None
        else:
            single_x = peak_xs[0]
            
            if single_x < vehicle_center:
                leftx_base = single_x
                left_detected = True
            else:
                rightx_base = single_x
                right_detected = True
        
        if left_detected:
            self.prev_left_x = leftx_base
        if right_detected:
            self.prev_right_x = rightx_base
        
        return leftx_base, rightx_base, left_detected, right_detected



    def detect_lanes_sliding_window(self, binary_warped):
        """Detect lanes using histogram-based sliding window"""
        bw = (binary_warped > 0).astype(np.uint8)
        bw255 = bw * 255
        
        histogram = np.sum(bw[(3*bw.shape[0])//4:, :], axis=0)
        
        out_img = cv2.cvtColor(bw255, cv2.COLOR_GRAY2BGR)
        
        vehicle_center = self.img_width // 2
        
        threshold = max(30, int(0.02 * np.max(histogram)))
        peaks = self.find_histogram_peaks(histogram, threshold, min_distance=80)
        
        leftx_base, rightx_base, left_detected, right_detected = \
            self.assign_lane_peaks(peaks, histogram, vehicle_center)
        
        leftx_current = leftx_base if left_detected else None
        rightx_current = rightx_base if right_detected else None
        
        window_height = np.int32(binary_warped.shape[0] // self.nwindows)
        
        nonzero = bw255.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            if leftx_current is not None:
                win_xleft_low = leftx_current - self.margin
                win_xleft_high = leftx_current + self.margin
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                 (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                
                if len(good_left_inds) > self.minpix:
                    leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            
            if rightx_current is not None:
                win_xright_low = rightx_current - self.margin
                win_xright_high = rightx_current + self.margin
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
                
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)
                
                if len(good_right_inds) > self.minpix:
                    rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([])
        
        left_fit = None
        right_fit = None
        
        if len(left_lane_inds) > 0:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            if len(leftx) > 3:
                left_fit = np.polyfit(lefty, leftx, 2)
                out_img[lefty, leftx] = [255, 0, 0]
        
        if len(right_lane_inds) > 0:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            if len(rightx) > 3:
                right_fit = np.polyfit(righty, rightx, 2)
                out_img[righty, rightx] = [0, 0, 255]
        
        left_fit, right_fit = self.reassign_lane_by_curvature(left_fit, right_fit)
        
        return left_fit, right_fit, out_img



    def reassign_lane_by_curvature(self, left_fit, right_fit):
        """곡률 기반 차선 재판별"""
        # ✅ 기본 임계값
        curvature_threshold = 0.0005
        
        # ✅ DETECT_PEDESTRIAN + 저속(0.3)일 때만 임계값 증가
        if self.mission_state == "DETECT_PEDESTRIAN" and self.base_speed <= 0.35:
            curvature_threshold = 0.003  # 0.0005 → 0.003 (6배 증가)
            rospy.loginfo_throttle(5.0, f"[Curvature] PEDESTRIAN low speed mode - threshold={curvature_threshold:.6f}")
        
        if left_fit is not None and right_fit is not None:
            return left_fit, right_fit
        
        if left_fit is None and right_fit is None:
            return None, None
        
        if left_fit is not None:
            a = left_fit[0]
            
            # ✅ 임계값보다 작으면 직선으로 간주 (재판별 안함)
            if abs(a) < curvature_threshold:
                rospy.loginfo_throttle(5.0, f"[Curvature] a={a:.6f} - 직선으로 판단 (threshold={curvature_threshold:.6f})")
                return left_fit, right_fit
            
            # 임계값보다 크면 곡선으로 판단하여 재판별
            if abs(a) >= curvature_threshold:
                if a < 0:
                    rospy.loginfo_throttle(1.0, f"[Curvature] a={a:.6f} < 0 → 오른쪽 차선으로 재할당")
                    return None, left_fit
            
            return left_fit, right_fit
        
        if right_fit is not None:
            a = right_fit[0]
            
            # ✅ 임계값보다 작으면 직선으로 간주 (재판별 안함)
            if abs(a) < curvature_threshold:
                rospy.loginfo_throttle(5.0, f"[Curvature] a={a:.6f} - 직선으로 판단 (threshold={curvature_threshold:.6f})")
                return left_fit, right_fit
            
            # 임계값보다 크면 곡선으로 판단하여 재판별
            if abs(a) >= curvature_threshold:
                if a > 0:
                    rospy.loginfo_throttle(1.0, f"[Curvature] a={a:.6f} > 0 → 왼쪽 차선으로 재할당")
                    return right_fit, None
            
            return left_fit, right_fit
        
        return left_fit, right_fit



    def generate_driving_path(self, left_fit, right_fit, img_shape):
        """Generate driving path from detected lanes"""
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        
        if self.avoid_side == "LEFT":
            if left_fit is not None:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                driving_fitx = left_fitx + self.avoid_lane_offset
                rospy.loginfo_throttle(2.0, f"[Lane] AVOID LEFT mode - using LEFT lane only (offset={self.avoid_lane_offset})")
            elif right_fit is not None:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                driving_fitx = right_fitx - self.avoid_lane_offset
                rospy.logwarn_throttle(2.0, "[Lane] AVOID LEFT but no left lane, using right")
            else:
                return None
            
            driving_path = np.column_stack((driving_fitx, ploty))
            return driving_path
        
        elif self.avoid_side == "RIGHT":
            if right_fit is not None:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                driving_fitx = right_fitx - self.avoid_lane_offset
                rospy.loginfo_throttle(2.0, f"[Lane] AVOID RIGHT mode - using RIGHT lane only (offset={self.avoid_lane_offset})")
            elif left_fit is not None:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                driving_fitx = left_fitx + self.avoid_lane_offset
                rospy.logwarn_throttle(2.0, "[Lane] AVOID RIGHT but no right lane, using left")
            else:
                return None
            
            driving_path = np.column_stack((driving_fitx, ploty))
            return driving_path
        
        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            driving_fitx = (left_fitx + right_fitx) / 2
        elif left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            driving_fitx = left_fitx + self.lane_offset
        elif right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            driving_fitx = right_fitx - self.lane_offset
        else:
            return None
        
        driving_path = np.column_stack((driving_fitx, ploty))
        return driving_path



    def pure_pursuit_control(self, driving_path, debug_img):
        """Pure Pursuit control with PD"""
        if driving_path is None or len(driving_path) == 0:
            return
        
        vehicle_x = self.img_width // 2
        vehicle_y = self.img_height
        
        look_ahead_idx = max(0, vehicle_y - self.look_ahead_distance)
        look_ahead_idx = min(look_ahead_idx, len(driving_path) - 1)
        
        target_point = driving_path[int(look_ahead_idx)]
        target_x = target_point[0]
        target_y = target_point[1]
        
        # Draw driving path
        for i in range(len(driving_path) - 1):
            pt1 = (int(driving_path[i][0]), int(driving_path[i][1]))
            pt2 = (int(driving_path[i+1][0]), int(driving_path[i+1][1]))
            cv2.line(debug_img, pt1, pt2, (0, 255, 255), 2)
        
        # Draw look-ahead point
        cv2.circle(debug_img, (int(target_x), int(target_y)), 10, (255, 0, 255), -1)
        cv2.circle(debug_img, (vehicle_x, vehicle_y), 10, (0, 255, 0), -1)
        
        # Calculate errors
        lateral_error = (target_x - vehicle_x) * self.xm_per_pix
        
        dx = target_x - vehicle_x
        dy = vehicle_y - target_y
        heading_error = np.arctan2(dx, dy + 1e-6)
        
        # PD control
        lateral_derivative = lateral_error - self.prev_lateral_error
        lateral_control = self.kp_lateral * lateral_error + self.kd_lateral * lateral_derivative
        
        heading_derivative = heading_error - self.prev_heading_error
        heading_control = self.kp_heading * heading_error + self.kd_heading * heading_derivative
        
        angular_vel = lateral_control + heading_control
        angular_vel = -1.0*np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        # Adjust speed
        speed = self.base_speed * (1.0 - 0.5 * abs(angular_vel) / self.max_angular_vel)
        
        # Publish
        self.publish_cmd(speed, angular_vel, mark_good=True)
        
        # Update errors
        self.prev_lateral_error = lateral_error
        self.prev_heading_error = heading_error
        
        # Debug info
        cv2.putText(debug_img, f"Lat Err: {lateral_error:.3f}m", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Head Err: {np.degrees(heading_error):.1f}deg", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Speed: {speed:.2f} (base:{self.base_speed:.2f})", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Angular: {angular_vel:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"LA: {self.look_ahead_distance}px", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)



    def run(self):
        """Run the node"""
        rospy.spin()



def main():
    try:
        node = LaneDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass



if __name__ == '__main__':
    main()
