#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roundabout Navigation Node for LIMO Pro (Simplified)
- State Manager 연동: ROTARY state일 때만 동작
- 장애물 감지 시 lane_state로 STOP 발행
- 장애물 없으면 lane_state로 LANE_FOLLOWING 발행
- cmd_vel 직접 제어 없음 (lane_detection에 위임)
"""

import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import LaserScan, Imu, CompressedImage, Image
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion


class RoundaboutState:
    IDLE = "IDLE"                    # 대기 (state가 ROTARY가 아닐 때)
    WAITING = "WAITING"              # 장애물 대기 (STOP 발행)
    DRIVING = "DRIVING"              # 주행 중 (LANE_FOLLOWING 발행)


class RoundaboutNode:
    def __init__(self):
        rospy.init_node('roundabout_node')
        
        # === State Manager 연동 ===
        self.current_mission_state = None  # state_manager로부터 받는 미션 상태
        self.enabled = False  # ROTARY state일 때만 True
        
        # === LiDAR 장애물 감지 파라미터 ===
        self.detect_x_min = rospy.get_param('~detect_x_min', 0.3)   # 전방 최소 거리 (m)
        self.detect_x_max = rospy.get_param('~detect_x_max', 0.8)   # 전방 최대 거리 (m)
        self.detect_y_min = rospy.get_param('~detect_y_min', -0.02)  # 좌우 범위 (m)
        self.detect_y_max = rospy.get_param('~detect_y_max', 0.3)  # 매우 좁은 오른쪽 범위 (완전 오른쪽 무시)
        self.obstacle_threshold = rospy.get_param('~obstacle_threshold', 3)  # 장애물 판단 포인트 수
        
        # === 내부 상태 ===
        self.state = RoundaboutState.IDLE
        
        # === 장애물 감지 ===
        self.obstacle_detected = False
        self.obstacle_points = []
        self.scan_data = None
        
        # === IMU (표시용) ===
        self.current_yaw = 0.0
        
        # === 차선 상태 (표시용) ===
        self.left_lane_x = -1
        self.right_lane_x = -1
        self.lane_steering = 0.0
        self.left_lane_detected = True
        self.right_lane_detected = True
        
        # === 이미지 처리 ===
        self.bridge = CvBridge()
        self.current_image = None
        self.img_width = 640
        self.img_height = 480
        
        # === Publishers ===
        # Lane State 발행 (STOP / LANE_FOLLOWING)
        self.pub_lane_state = rospy.Publisher('/state_manager/lane_state', String, queue_size=1)
        
        # 디버그용
        self.pub_state = rospy.Publisher('/limo/roundabout_state', String, queue_size=1)
        self.pub_debug_image = rospy.Publisher('/limo/roundabout/debug', Image, queue_size=1)
        self.pub_debug_compressed = rospy.Publisher('/limo/roundabout/debug/compressed', CompressedImage, queue_size=1)
        
        # === Subscribers ===
        # State Manager 구독
        self.sub_mission_state = rospy.Subscriber('/state_manager/state', String, self.mission_state_callback, queue_size=1)
        
        # 센서
        self.sub_image = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.sub_imu = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        
        # 차선 정보 (표시용)
        self.sub_lane_points = rospy.Subscriber('/limo/lane_points', Point, self.lane_points_callback, queue_size=1)
        self.sub_steering = rospy.Subscriber('/limo/steering_offset', Float32, self.steering_callback, queue_size=1)
        
        # Timer for control loop (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Roundabout Node Initialized")
        rospy.loginfo("Waiting for state: ROTARY")
        rospy.loginfo("View: rqt_image_view /limo/roundabout/debug")
        rospy.loginfo("=" * 50)
    
    # ==================== Callbacks ====================
    
    def mission_state_callback(self, msg):
        """State Manager로부터 미션 상태 수신"""
        self.current_mission_state = msg.data
        self.enabled = (self.current_mission_state == "ROTARY")
        
        if self.enabled:
            rospy.loginfo("Roundabout: ENABLED (ROTARY state)")
            self.state = RoundaboutState.DRIVING
        else:
            if self.state != RoundaboutState.IDLE:
                rospy.loginfo(f"Roundabout: DISABLED (state: {self.current_mission_state})")
            self.state = RoundaboutState.IDLE

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
        """차선 좌표 수신 (표시용)"""
        self.left_lane_x = int(msg.x)
        self.right_lane_x = int(msg.y)
        self.left_lane_detected = self.left_lane_x > 0
        self.right_lane_detected = self.right_lane_x > 0
    
    def steering_callback(self, msg):
        """조향값 수신 (감지 영역 회전에 사용)"""
        self.lane_steering = msg.data
    
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
        """LiDAR 데이터로 전방 장애물 감지 (조향각 기반 영역 회전)"""
        if self.scan_data is None:
            return
        
        self.obstacle_points = []
        angle = self.scan_data.angle_min
        
        # 조향각을 회전 각도로 변환 (steering이 양수면 좌회전)
        # steering 값은 보통 -1.0 ~ 1.0 범위, 적당히 스케일링
        steering_angle = self.lane_steering * 0.5  # 라디안으로 스케일링 (조절 가능)
        
        cos_s = math.cos(steering_angle)
        sin_s = math.sin(steering_angle)
        
        for r in self.scan_data.ranges:
            if self.scan_data.range_min < r < self.scan_data.range_max:
                # 극좌표 → 직교좌표 (x: 전방, y: 좌측 양수)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                
                # 조향 방향으로 좌표 회전 (역회전: 감지 영역을 조향 방향으로 회전)
                # 포인트를 역방향으로 회전시켜서 직진 기준 좌표계로 변환
                x_rot = x * cos_s + y * sin_s
                y_rot = -x * sin_s + y * cos_s
                
                # 회전된 좌표계에서 감지 영역 체크
                if (self.detect_x_min <= x_rot <= self.detect_x_max and
                    self.detect_y_min <= y_rot <= self.detect_y_max):
                    self.obstacle_points.append((x, y))  # 원본 좌표 저장 (시각화용)
            
            angle += self.scan_data.angle_increment
        
        self.obstacle_detected = len(self.obstacle_points) >= self.obstacle_threshold
    
    # ==================== Control Loop ====================
    
    def control_loop(self, event):
        """메인 제어 루프 - lane_state 발행만"""
        # 항상 디버그 이미지 발행
        self.publish_debug_image()
        
        # 상태 발행 (디버그용)
        self.pub_state.publish(String(data=self.state))
        
        # ROTARY state가 아니면 동작 안함
        if not self.enabled:
            return
        
        # === 장애물 감지에 따라 lane_state 발행 ===
        if self.obstacle_detected:
            # 장애물 발견 → STOP
            if self.state != RoundaboutState.WAITING:
                self.state = RoundaboutState.WAITING
                rospy.loginfo("Roundabout: 장애물 감지 - STOP 발행")
            self.pub_lane_state.publish(String(data="STOP"))
        else:
            # 장애물 없음 → LANE_FOLLOWING
            if self.state != RoundaboutState.DRIVING:
                self.state = RoundaboutState.DRIVING
                rospy.loginfo("Roundabout: 장애물 없음 - LANE_FOLLOWING 발행")
            self.pub_lane_state.publish(String(data="LANE_FOLLOWING"))
    
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
        cv2.rectangle(vis_img, (0, 0), (w, 60), (0, 0, 0), -1)
        status_text = f"ROUNDABOUT: {self.state}"
        mission_text = f"Mission: {self.current_mission_state}"
        enabled_text = "ENABLED" if self.enabled else "DISABLED"
        
        cv2.putText(vis_img, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis_img, f"{mission_text} ({enabled_text})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
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
        
        # Lane State 표시
        lane_state_text = "STOP" if self.obstacle_detected else "LANE_FOLLOWING"
        lane_state_color = (0, 0, 255) if self.obstacle_detected else (0, 255, 0)
        cv2.putText(vis_img, f"LaneState: {lane_state_text}", (obs_x + 5, info_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lane_state_color, 1)
        
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
        """LiDAR 감지 영역 미니맵 (회전된 감지 영역 표시)"""
        cv2.rectangle(img, (x, y), (x + width, y + height), (30, 30, 30), -1)
        
        cx = x + width // 2
        cy = y + height - 3
        scale = 80  # 스케일 팩터
        
        # 조향각
        steering_angle = self.lane_steering * 0.5
        cos_s = math.cos(-steering_angle)  # 표시용은 반대로
        sin_s = math.sin(-steering_angle)
        
        # 회전된 감지 영역 사각형 그리기 (녹색/노란색)
        corners = [
            (self.detect_x_min, self.detect_y_min),
            (self.detect_x_min, self.detect_y_max),
            (self.detect_x_max, self.detect_y_max),
            (self.detect_x_max, self.detect_y_min),
        ]
        
        rotated_corners = []
        for px, py in corners:
            # 회전 적용
            rx = px * cos_s - py * sin_s
            ry = px * sin_s + py * cos_s
            # 미니맵 좌표로 변환
            mx = int(cx - ry * scale)
            my = int(cy - (rx - self.detect_x_min) * 50)
            rotated_corners.append((mx, my))
        
        # 감지 영역 다각형 그리기
        pts = np.array(rotated_corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 1)  # 노란색 테두리
        
        # 장애물 포인트 (빨간색)
        for px, py in self.obstacle_points:
            mx = int(cx - py * scale)
            my = int(cy - (px - self.detect_x_min) * 50)
            if x <= mx <= x + width and y <= my <= y + height:
                cv2.circle(img, (mx, my), 2, (0, 0, 255), -1)
        
        # 로봇 위치 (녹색)
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = RoundaboutNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
