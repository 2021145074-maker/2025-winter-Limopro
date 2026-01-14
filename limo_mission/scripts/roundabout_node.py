#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roundabout Navigation Node for LIMO Pro (Simplified)
- launch 시 바로 장애물 확인 후 라인트레이싱
- lane_detect_node의 steering 사용 (별도 튜닝 가능)
- IMU는 디버그 표시만
- cmd_vel 직접 발행
"""

import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import LaserScan, Imu, CompressedImage, Image
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Point
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client as DynClient
from limo_mission.cfg import RoundaboutConfig


class RoundaboutState:
    IDLE = "IDLE"                    # 대기 (트리거 대기)
    WAITING = "WAITING"              # 장애물 대기
    DRIVING = "DRIVING"              # 라인트레이싱 주행 중


class RoundaboutNode:
    def __init__(self):
        rospy.init_node('roundabout_node')
        
        # === LiDAR 장애물 감지 파라미터 ===
        self.detect_x_min = 0.2      # 전방 최소 거리 (m)
        self.detect_x_max = 0.8      # 전방 최대 거리 (m)
        self.detect_y_min = -0.3     # 좌우 범위 (m)
        self.detect_y_max = 0.3
        self.obstacle_threshold = 3   # 장애물 판단 포인트 수
        
        # === 속도 파라미터 (lane_detect_node에서 수신) ===
        self.lane_speed = 0.25       # 기본값, lane_detect_node에서 수신
        self.steering_gain = 1.0      # 회전교차로용 조향 게인 (민감하게)
        
        # === Roundabout 전용 lane_detect 튜닝 값 (여기서 수정!) ===
        self.roundabout_lane_params = {
            'kp': 0.015,              # 더 민감한 P 게인
            'kd': 0.008,              # 더 민감한 D 게인
            'base_speed': 0.2,        # 회전교차로 속도
            'lane_offset': 160,       # 차선 오프셋
            'roi_top': 280,           # ROI 상단
            'roi_bottom': 420,        # ROI 하단
        }
        self.original_lane_params = None  # 원래 파라미터 저장용
        self.lane_client = None       # dynamic_reconfigure client
        
        # === 상태 ===
        self.state = RoundaboutState.DRIVING
        self.active = True
        
        # === 장애물 감지 ===
        self.obstacle_detected = False
        self.obstacle_points = []
        self.scan_data = None
        
        # === IMU (표시만) ===
        self.current_yaw = 0.0
        
        # === 차선 상태 (lane_detect_node에서 수신) ===
        self.left_lane_x = -1
        self.right_lane_x = -1
        self.lane_steering = 0.0
        self.lane_speed = 0.25  # lane_detect_node에서 수신할 속도
        self.left_lane_detected = True
        self.right_lane_detected = True
        
        # === 이미지 처리 ===
        self.bridge = CvBridge()
        self.current_image = None
        self.img_width = 640
        self.img_height = 480
        
        # Publishers
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_state = rospy.Publisher('/limo/roundabout_state', String, queue_size=1)
        self.pub_active = rospy.Publisher('/limo/roundabout_active', Bool, queue_size=1)
        self.pub_done = rospy.Publisher('/limo/roundabout_done', Bool, queue_size=1)
        self.pub_lane_mode = rospy.Publisher('/limo/lane_mode', String, queue_size=1)
        
        # Debug Image Publishers
        self.pub_debug_image = rospy.Publisher('/limo/roundabout/debug', Image, queue_size=1)
        self.pub_debug_compressed = rospy.Publisher('/limo/roundabout/debug/compressed', CompressedImage, queue_size=1)
        
        # Subscribers
        self.sub_image = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.sub_imu = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sub_lane_points = rospy.Subscriber('/limo/lane_points', Point, self.lane_points_callback, queue_size=1)
        self.sub_steering = rospy.Subscriber('/limo/steering_offset', Float32, self.steering_callback, queue_size=1)
        self.sub_speed = rospy.Subscriber('/limo/lane_speed', Float32, self.speed_callback, queue_size=1)
        
        # Dynamic reconfigure
        self.srv = Server(RoundaboutConfig, self.reconfigure_callback)
        
        # lane_detect dynamic_reconfigure client 연결 시도
        self.setup_lane_client()
        
        # 시작 시 roundabout 파라미터 적용
        self.apply_roundabout_params()
        
        # Timer for control loop (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Roundabout Node (Auto Start)")
        rospy.loginfo("View: rqt_image_view /limo/roundabout/debug")
        rospy.loginfo("=" * 50)
    
    def reconfigure_callback(self, config, level):
        """Dynamic reconfigure"""
        self.detect_x_min = config.detect_x_min
        self.detect_x_max = config.detect_x_max
        self.obstacle_threshold = config.obstacle_threshold
        # roundabout_speed는 제거 - lane_detect_node의 base_speed로 관리
        self.steering_gain = config.steering_gain
        rospy.loginfo(f"Reconfigure: steering_gain={self.steering_gain:.2f}")
        return config
    
    def setup_lane_client(self):
        """lane_detect_node의 dynamic_reconfigure client 설정"""
        try:
            self.lane_client = DynClient('/lane_detect_node', timeout=2.0)
            rospy.loginfo("Connected to lane_detect_node dynamic_reconfigure")
        except Exception as e:
            rospy.logwarn(f"Could not connect to lane_detect_node: {e}")
            self.lane_client = None
    
    def apply_roundabout_params(self):
        """Roundabout용 lane_detect 파라미터 적용"""
        if self.lane_client is None:
            self.setup_lane_client()
        
        if self.lane_client is not None:
            try:
                # 현재 파라미터 저장
                current = self.lane_client.get_configuration()
                self.original_lane_params = {
                    'kp': current['kp'],
                    'kd': current['kd'],
                    'base_speed': current['base_speed'],
                    'lane_offset': current['lane_offset'],
                    'roi_top': current['roi_top'],
                    'roi_bottom': current['roi_bottom'],
                }
                rospy.loginfo(f"Saved original params: {self.original_lane_params}")
                
                # Roundabout 파라미터 적용
                self.lane_client.update_configuration(self.roundabout_lane_params)
                rospy.loginfo(f"Applied roundabout params: {self.roundabout_lane_params}")
            except Exception as e:
                rospy.logwarn(f"Failed to apply roundabout params: {e}")
    
    def restore_lane_params(self):
        """원래 lane_detect 파라미터 복구"""
        if self.lane_client is not None and self.original_lane_params is not None:
            try:
                self.lane_client.update_configuration(self.original_lane_params)
                rospy.loginfo(f"Restored original params: {self.original_lane_params}")
            except Exception as e:
                rospy.logwarn(f"Failed to restore params: {e}")
    
    # ==================== Callbacks ====================

    def scan_callback(self, msg):
        """LiDAR 스캔 콜백"""
        self.scan_data = msg
        self.process_lidar()
    
    def imu_callback(self, msg):
        """IMU 콜백 - YAW 표시용"""
        orientation = msg.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        self.current_yaw = math.degrees(yaw)
    
    def lane_points_callback(self, msg):
        """lane_detect_node에서 차선 좌표 수신"""
        self.left_lane_x = int(msg.x)
        self.right_lane_x = int(msg.y)
        self.left_lane_detected = self.left_lane_x > 0
        self.right_lane_detected = self.right_lane_x > 0
    
    def steering_callback(self, msg):
        """lane_detect_node에서 조향값 수신"""
        self.lane_steering = msg.data
    
    def speed_callback(self, msg):
        """lane_detect_node에서 속도값 수신"""
        self.lane_speed = msg.data
    
    def image_callback(self, msg):
        """카메라 이미지 수신"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if self.current_image is not None:
                self.img_height, self.img_width = self.current_image.shape[:2]
        except Exception as e:
            rospy.logwarn_throttle(5, f"Image decode error: {e}")
    
    # ==================== LiDAR Processing ====================
    
    def process_lidar(self):
        """LiDAR 데이터로 전방 장애물 감지"""
        if self.scan_data is None:
            return
        
        self.obstacle_points = []
        angle = self.scan_data.angle_min
        
        for r in self.scan_data.ranges:
            if self.scan_data.range_min < r < self.scan_data.range_max:
                # 극좌표 → 직교좌표 (x: 전방, y: 좌측 양수)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                
                # 감지 영역 내 포인트 필터링 (전방 x > 0)
                if (self.detect_x_min <= x <= self.detect_x_max and
                    self.detect_y_min <= y <= self.detect_y_max):
                    self.obstacle_points.append((x, y))
            
            angle += self.scan_data.angle_increment
        
        self.obstacle_detected = len(self.obstacle_points) >= self.obstacle_threshold
        
        # 디버그 로그
        if self.obstacle_detected:
            rospy.loginfo_throttle(1, f"Obstacle detected: {len(self.obstacle_points)} points")
    
    # ==================== Control Loop ====================
    
    def control_loop(self, event):
        """메인 제어 루프 - 단순화"""
        # 항상 디버그 이미지 발행
        self.publish_debug_image()
        
        # 상태 발행
        self.pub_state.publish(String(data=self.state))
        self.pub_active.publish(Bool(data=self.active))
        
        if not self.active:
            return
        
        cmd = Twist()
        
        # === WAITING: 장애물 대기 ===
        if self.state == RoundaboutState.WAITING:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
            if not self.obstacle_detected:
                self.state = RoundaboutState.DRIVING
                rospy.loginfo("Roundabout: 장애물 없음 - DRIVING")
        
        # === DRIVING: 라인트레이싱 주행 ===
        elif self.state == RoundaboutState.DRIVING:
            # 주행 중 장애물 발견하면 정지
            if self.obstacle_detected:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.state = RoundaboutState.WAITING
                rospy.loginfo("Roundabout: 장애물 감지 - WAITING")
            else:
                # 라인트레이싱 (lane_detect_node의 속도 사용)
                cmd.linear.x = self.lane_speed
                cmd.angular.z = -self.lane_steering * self.steering_gain
        
        # cmd_vel 발행
        self.pub_cmd.publish(cmd)
    
    # ==================== Debug Image ====================
    
    def publish_debug_image(self):
        """상태 정보를 오버레이한 디버그 이미지 발행"""
        if self.current_image is None:
            return
        
        vis_img = self.current_image.copy()
        h, w = vis_img.shape[:2]
        
        # 상태별 색상 정의
        state_colors = {
            RoundaboutState.IDLE: (128, 128, 128),      # 회색
            RoundaboutState.WAITING: (0, 0, 255),       # 빨간색
            RoundaboutState.DRIVING: (0, 255, 0),       # 녹색
        }
        color = state_colors.get(self.state, (255, 255, 255))
        
        # 상태 배너 (상단)
        cv2.rectangle(vis_img, (0, 0), (w, 50), (0, 0, 0), -1)
        status_text = f"ROUNDABOUT: {self.state}"
        if not self.active:
            status_text += " (Waiting Trigger)"
        cv2.putText(vis_img, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # 차선 감지 상태 (하단 왼쪽)
        info_y = h - 100
        cv2.rectangle(vis_img, (0, info_y), (220, h), (0, 0, 0), -1)
        
        left_color = (0, 255, 0) if self.left_lane_detected else (0, 0, 255)
        right_color = (0, 255, 0) if self.right_lane_detected else (0, 0, 255)
        
        cv2.putText(vis_img, f"Left: {'O' if self.left_lane_detected else 'X'}", (10, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
        cv2.putText(vis_img, f"Right: {'O' if self.right_lane_detected else 'X'}", (110, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
        cv2.putText(vis_img, f"Steering: {self.lane_steering:.3f}", (10, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_img, f"YAW: {self.current_yaw:.1f} deg", (10, info_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 장애물 상태 (하단 오른쪽)
        obs_x = w - 180
        cv2.rectangle(vis_img, (obs_x, info_y), (w, h), (0, 0, 0), -1)
        
        obs_color = (0, 0, 255) if self.obstacle_detected else (0, 255, 0)
        obs_text = "BLOCKED" if self.obstacle_detected else "CLEAR"
        cv2.putText(vis_img, f"Obstacle: {obs_text}", (obs_x + 5, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, obs_color, 2)
        cv2.putText(vis_img, f"Points: {len(self.obstacle_points)}", (obs_x + 5, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # LiDAR 미니맵
        self.draw_lidar_minimap(vis_img, obs_x + 5, info_y + 55, 170, 40)
        
        # 이미지 발행
        try:
            self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            
            _, compressed = cv2.imencode('.jpg', vis_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = compressed.tobytes()
            self.pub_debug_compressed.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5, f"Debug image error: {e}")
    
    def draw_lidar_minimap(self, img, x, y, width, height):
        """LiDAR 감지 영역 미니맵"""
        cv2.rectangle(img, (x, y), (x + width, y + height), (30, 30, 30), -1)
        
        cx = x + width // 2
        cy = y + height - 3
        
        # 장애물 포인트
        for px, py in self.obstacle_points:
            mx = int(cx - py * 80)
            my = int(cy - (px - self.detect_x_min) * 40)
            if x <= mx <= x + width and y <= my <= y + height:
                cv2.circle(img, (mx, my), 2, (0, 0, 255), -1)
        
        # 로봇 위치
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = RoundaboutNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
