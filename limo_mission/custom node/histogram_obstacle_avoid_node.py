#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rospy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN


class ObstacleAvoidNode:
    def __init__(self):
        rospy.init_node('dbscan_obstacle_avoid_node')
        
        # --- 고정 파라미터 (Config/Yaml 대신 코드 내에서 관리) ---
        self.safe_distance = 1.4    # 장애물 인지 시작 거리 (x_min 기준)
        self.stop_distance = 0.4    # 비상 정지 거리
        self.avoid_speed = 0.2      # 회피 주행 속도
        self.lane_speed = 0.5      # 일반 주행 속도
        self.max_steering = 0.5     # 최대 조향 제한
        
        # PD 제어 게인 (실험을 통해 튜닝 필요)
        self.kp = 1.2
        self.kd = 0.5
        self.prev_error = 0.0
        
        # 상태 변수
        self.current_state = ""
        self.last_log_state = ""
        self.lane_steering = 0.0
        self.ranges = None
        self.angle_increment = 0
        
        # 회피 상태 관리
        self.is_avoiding = False           # 현재 회피 중인지
        self.locked_target_y = None        # 회피 시작 시 고정된 target_y
        self.locked_target_x = None        # 회피 시작 시 고정된 target_x (장애물 앞)
        self.locked_obs_y_min = None       # 회피 시작 시 장애물 y_min (종료 조건용)
        self.locked_obs_y_max = None       # 회피 시작 시 장애물 y_max (종료 조건용)
        self.avoid_start_time = None       # 회피 시작 시간
        self.min_avoid_duration = 1.5      # 최소 회피 시간 (초) - 이 시간 전에는 종료 불가
        self.clear_count = 0               # 장애물 통과 확인 카운터
        self.clear_threshold = 15          # 연속 N회 clear 시 회피 종료
        self.avoid_direction = "NONE"      # 회피 방향: "LEFT", "RIGHT", "NONE"
        
        # === 초기 직진 설정 ===
        self.initial_straight_duration = 2.0  # 모드 진입 후 직진 시간 (초) - 2.0 → 0.5로 단축
        self.mode_entry_time = None           # 모드 진입 시간
        self.initial_straight_speed = 0.3     # 초기 직진 속도
        self.prev_state = None                # 이전 상태 추적
        
        # 시각화용 변수
        self.bridge = CvBridge()
        self.vis_size = 500
        self.last_points = np.array([])
        self.last_labels = np.array([])
        self.last_occupied = [False, False, False]
        self.last_target_obs = None
        self.last_target_y = 0.0
        
        # --- ROS 통신 설정 ---
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_lane_state = rospy.Publisher('/state_manager/lane_state', String, queue_size=1)
        self.pub_debug_image = rospy.Publisher('/limo/obstacle/debug', Image, queue_size=1)
        # 회피 방향 발행: "LEFT", "RIGHT", "NONE"
        self.pub_avoid_side = rospy.Publisher('/limo/obstacle/avoid_side', String, queue_size=1)
        
        rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber('/limo/steering_offset', Float32, self.steering_callback, queue_size=1)
        rospy.Subscriber('/state_manager/state', String, self.state_callback, queue_size=1)

        # 20Hz 주기로 제어 루프 실행
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        self.vis_timer = rospy.Timer(rospy.Duration(0.1), self.visualization_callback)
        
        rospy.loginfo("="*50)
        rospy.loginfo("DBSCAN Obstacle Avoidance Node Initialized")
        rospy.loginfo("Algorithm: 3-Sector Occupancy + PD Control")
        rospy.loginfo(f"Initial straight duration: {self.initial_straight_duration}s")
        rospy.loginfo("Debug: rqt_image_view /limo/obstacle/debug")
        rospy.loginfo("="*50)


    def state_callback(self, msg):
        prev_state = self.current_state
        new_state = msg.data
        
        # OBSTACLE_AVOIDANCE 모드 진입 감지
        if new_state == "OBSTACLE_AVOIDANCE" and prev_state != "OBSTACLE_AVOIDANCE":
            self.mode_entry_time = rospy.Time.now()
            rospy.loginfo(f"[OBSTACLE] Mode entered, starting {self.initial_straight_duration}s straight")
        
        # OBSTACLE_AVOIDANCE 모드 종료 시 avoid_side 리셋
        if prev_state == "OBSTACLE_AVOIDANCE" and new_state != "OBSTACLE_AVOIDANCE":
            self.avoid_direction = "NONE"
            self.pub_avoid_side.publish(String(data="NONE"))
            rospy.loginfo(f"[OBSTACLE] Mode exited, reset avoid_side to NONE")
        
        self.prev_state = prev_state
        self.current_state = new_state


    def steering_callback(self, msg):
        self.lane_steering = msg.data


    def scan_callback(self, msg):
        self.ranges = np.array(msg.ranges)
        self.angle_increment = msg.angle_increment


    def _get_cartesian_points(self):
        """라이다 극좌표 데이터를 DBSCAN용 XY 직교 좌표로 변환 (정면 범위)"""
        if self.ranges is None: return np.array([])
        
        points = []
        total_pts = len(self.ranges)
        center_idx = total_pts // 2
        # 연산 효율을 위해 정면 ±45도 범위 데이터만 사용
        limit_idx = int(np.radians(45) / self.angle_increment)
        
        for i in range(center_idx - limit_idx, center_idx + limit_idx):
            if i < 0 or i >= total_pts: continue
            dist = self.ranges[i]
            if 0.1 < dist < 2.5: # 2.5m 이내의 유효한 데이터만 필터링
                angle = (i - center_idx) * self.angle_increment
                # 로봇 정면이 X축, 좌측이 Y축인 좌표계
                points.append([dist * np.cos(angle), dist * np.sin(angle)])
        return np.array(points)


    def _analyze_obstacle(self, points):
        """DBSCAN 클러스터링을 통해 장애물의 경계 상자(Bounding Box) 추출"""
        if len(points) < 5: 
            self.last_labels = np.array([])
            return None, [False, False, False]
        
        # eps: 군집화 반경(20cm), min_samples: 최소 포인트 개수
        db = DBSCAN(eps=0.2, min_samples=4).fit(points)
        labels = db.labels_
        self.last_labels = labels  # 시각화용 저장
        
        occupied = [False, False, False] # [Left, Center, Right] 점유 상태
        target_obs = None
        min_x = 10.0
        
        for label in set(labels):
            if label == -1: continue # 노이즈 포인트 무시
            cluster = points[labels == label]
            
            # 클러스터 통계치 계산
            c_x_min = np.min(cluster[:, 0])
            c_y_min = np.min(cluster[:, 1])
            c_y_max = np.max(cluster[:, 1])
            
            # 클러스터가 걸쳐있는 모든 섹터를 점유로 표시
            sector_threshold = 0.18  # 섹터 경계 (차량 폭에 맞춰 조정)
            
            if c_y_max > sector_threshold:  # 클러스터가 왼쪽 섹터(Y>0.18)까지 걸침
                occupied[0] = True
            if c_y_min < -sector_threshold:  # 클러스터가 오른쪽 섹터(Y<-0.18)까지 걸침
                occupied[2] = True
            if c_y_max > -sector_threshold and c_y_min < sector_threshold:  # 중앙 섹터에 걸침
                occupied[1] = True
            
            # 중앙 장애물 중 가장 가까운 것을 회피 대상으로 선정
            is_center_obstacle = (c_y_max > -sector_threshold and c_y_min < sector_threshold)
            if is_center_obstacle and c_x_min < min_x:
                min_x = c_x_min
                target_obs = {
                    'x_min': c_x_min,
                    'x_max': np.max(cluster[:, 0]),
                    'y_min': c_y_min,
                    'y_max': c_y_max
                }
        
        return target_obs, occupied


    def control_loop(self, event):
        # 상위 State Manager가 회피 모드일 때만 동작
        if self.current_state != "OBSTACLE_AVOIDANCE":
            # 상태가 바뀌면 회피 상태 리셋
            if self.is_avoiding:
                self.is_avoiding = False
                self.locked_target_y = None
                self.locked_target_x = None
                self.clear_count = 0
            
            # avoid_direction이 NONE이 아니면 리셋
            if self.avoid_direction != "NONE":
                self.avoid_direction = "NONE"
                self.pub_avoid_side.publish(String(data="NONE"))
            return
        
        cmd = Twist()
        
        # === 초기 직진 구간 ===
        if self.mode_entry_time is not None:
            elapsed = (rospy.Time.now() - self.mode_entry_time).to_sec()
            if elapsed < self.initial_straight_duration:
                # 초기 직진: 조향 없이 직진만
                cmd.linear.x = self.initial_straight_speed
                cmd.angular.z = 0.0
                state_str = f"INITIAL_STRAIGHT ({elapsed:.1f}/{self.initial_straight_duration}s)"
                
                # 차선 주행 비활성화
                self.pub_lane_state.publish(String(data="STOP"))
                
                if state_str != self.last_log_state:
                    rospy.loginfo(f"[OBSTACLE] {state_str}")
                    self.last_log_state = state_str
                
                self.pub_cmd.publish(cmd)
                return

        points = self._get_cartesian_points()
        self.last_points = points  # 시각화용 저장
        obs, occupied = self._analyze_obstacle(points)
        self.last_occupied = occupied  # 시각화용 저장
        self.last_target_obs = obs  # 시각화용 저장
        
        cmd = Twist()
        state_str = ""
        
        # 장애물 감지 여부
        obstacle_detected = obs and obs['x_min'] < self.safe_distance
        
        # ========== 1. 회피 시작 조건 ==========
        if obstacle_detected and not self.is_avoiding:
            # 회피 시작! target 고정
            self.is_avoiding = True
            self.clear_count = 0
            self.avoid_start_time = rospy.Time.now()
            
            # [핵심] target_y 고정 (장애물 끝 + 마진)
            margin_y = 0.45  # Y 방향 안전 마진
            if not occupied[0]:  # 왼쪽이 비었으면 왼쪽으로
                self.locked_target_y = obs['y_max'] + margin_y
                self.avoid_direction = "LEFT"
            elif not occupied[2]:  # 오른쪽이 비었으면 오른쪽으로
                self.locked_target_y = obs['y_min'] - margin_y
                self.avoid_direction = "RIGHT"
            else:  # 양쪽 막힘 → 더 먼 쪽으로
                self.locked_target_y = 0.6 if abs(obs['y_min']) < abs(obs['y_max']) else -0.6
                self.avoid_direction = "LEFT" if self.locked_target_y > 0 else "RIGHT"
            
            # [핵심] target_x를 장애물 앞쪽으로 설정 (장애물 x_max 뒤로 지나가야 회피 완료)
            margin_x = -0.3  # X 방향 마진 (장애물 앞으로)
            self.locked_target_x = obs['x_max'] + margin_x
            
            # 장애물 위치 저장 (종료 조건용)
            self.locked_obs_y_min = obs['y_min']
            self.locked_obs_y_max = obs['y_max']
            
            # 차선 주행 비활성화
            self.pub_lane_state.publish(String(data="STOP"))
            rospy.loginfo(f"[OBSTACLE] AVOID START! target=({self.locked_target_x:.2f}, {self.locked_target_y:.2f})")
        
        # ========== 2. 회피 중 ==========
        if self.is_avoiding:
            # 비상 정지: 너무 가까움
            if obs and obs['x_min'] < self.stop_distance:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                state_str = "EMERGENCY_STOP"
                self.last_target_y = 0.0
                self.pub_lane_state.publish(String(data="STOP"))
            else:
                # 고정된 target으로 PD 제어
                target_y = self.locked_target_y
                target_x = self.locked_target_x
                self.last_target_y = target_y
                
                # PD 제어: 고정된 목표점을 향해 조향
                target_angle = np.arctan2(target_y, target_x)
                error = target_angle
                steering = (self.kp * error) + (self.kd * (error - self.prev_error))
                self.prev_error = error
                
                cmd.linear.x = self.avoid_speed
                cmd.angular.z = np.clip(steering, -self.max_steering, self.max_steering)
                state_str = f"AVOIDING (target: x={target_x:.2f}, y={target_y:.2f})"
                
                # 차선 주행 비활성화 유지 및 회피 방향 발행
                self.pub_lane_state.publish(String(data="STOP"))
                self.pub_avoid_side.publish(String(data=self.avoid_direction))
            
            # ========== 3. 회피 종료 조건 ==========
            # 최소 회피 시간 체크
            elapsed_time = (rospy.Time.now() - self.avoid_start_time).to_sec()
            min_time_passed = elapsed_time >= self.min_avoid_duration
            
            # 조건 1: 전방에 장애물이 없음
            front_clear = (not obs) or (obs['x_min'] > self.safe_distance)
            
            # 조건 2: 중앙 섹터가 비어있음
            center_clear = not occupied[1]
            
            # 조건 3: 원래 장애물을 옆으로 지나쳤는지 확인
            passed_obstacle = False
            if self.locked_obs_y_min is not None and self.locked_obs_y_max is not None:
                if self.locked_target_y > 0:  # 왼쪽으로 회피
                    # 왼쪽에 있던 장애물이 이제 오른쪽에 있거나 없어야 함
                    if obs is None or obs['y_max'] < 0:
                        passed_obstacle = True
                else:  # 오른쪽으로 회피
                    # 오른쪽에 있던 장애물이 이제 왼쪽에 있거나 없어야 함
                    if obs is None or obs['y_min'] > 0:
                        passed_obstacle = True
            
            # 종료 조건: 최소 시간 경과 AND (전방 클리어 OR 장애물 통과)
            can_finish = min_time_passed and (front_clear or (center_clear and passed_obstacle))
            
            if can_finish:
                self.clear_count += 1
                state_str = f"AVOIDING (clearing: {self.clear_count}/{self.clear_threshold}, t={elapsed_time:.1f}s)"
                if self.clear_count >= self.clear_threshold:
                    # 회피 완료!
                    self.is_avoiding = False
                    self.locked_target_y = None
                    self.locked_target_x = None
                    self.locked_obs_y_min = None
                    self.locked_obs_y_max = None
                    self.clear_count = 0
                    self.prev_error = 0.0
                    
                    # 차선 주행 다시 활성화 (avoid_side는 유지)
                    self.pub_lane_state.publish(String(data="LANE_FOLLOWING"))
                    state_str = f"AVOID_COMPLETE ({elapsed_time:.1f}s) -> LANE_FOLLOWING (keep {self.avoid_direction} lane)"
                    rospy.loginfo(f"[OBSTACLE] {state_str}")
            else:
                self.clear_count = 0  # 아직 조건 불충족, 카운터 리셋
                if not min_time_passed:
                    state_str = f"AVOIDING (min_time: {elapsed_time:.1f}/{self.min_avoid_duration}s)"
            
            # ✅ 회피 중일 때만 cmd_vel 발행
            self.pub_cmd.publish(cmd)
        
        # ========== 4. 회피 모드 아님 (장애물 없음) ==========
        else:
            # 차선 추종 유지 (avoid_direction이 설정되어 있으면 유지)
            self.pub_lane_state.publish(String(data="LANE_FOLLOWING"))
            
            # 이전에 회피했던 방향이 있으면 계속 유지
            if self.avoid_direction != "NONE":
                self.pub_avoid_side.publish(String(data=self.avoid_direction))
                state_str = f"CLEAR (keeping {self.avoid_direction} lane)"
            else:
                self.pub_avoid_side.publish(String(data="NONE"))
                state_str = "CLEAR"
            
            self.prev_error = 0.0
            self.last_target_y = 0.0
            
            # ✅ CLEAR 상태에서는 cmd_vel 발행 안함 (lane_following이 제어)
            # self.pub_cmd.publish(cmd) ← 삭제!
        
        # 로깅
        if state_str != self.last_log_state:
            rospy.loginfo(f"[OBSTACLE] {state_str}")
            self.last_log_state = state_str


    def visualization_callback(self, event):
        """디버그 시각화 이미지 생성 및 퍼블리시"""
        self.publish_debug_image()


    def publish_debug_image(self):
        """LiDAR 포인트, DBSCAN 클러스터, 3섹터 점유, 회피 방향 시각화"""
        size = self.vis_size
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # 좌표 변환: 로봇 위치는 하단 중앙
        cx, cy = size // 2, size - 80
        scale = 150  # 1m = 150 pixels
        
        # ===== 1. 거리 그리드 =====
        for r_m in [0.5, 1.0, 1.5, 2.0]:
            r_px = int(r_m * scale)
            cv2.circle(img, (cx, cy), r_px, (40, 40, 40), 1)
            cv2.putText(img, f"{r_m}m", (cx + r_px - 20, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        
        # ===== 2. 3섹터 경계선 =====
        sector_boundary = int(0.18 * scale)
        cv2.line(img, (cx - sector_boundary, 0), (cx - sector_boundary, size), (60, 60, 60), 1)
        cv2.line(img, (cx + sector_boundary, 0), (cx + sector_boundary, size), (60, 60, 60), 1)
        
        # ===== 3. 섹터 점유 상태 표시 (상단 바) =====
        bar_width = size // 3
        bar_height = 35
        sector_names = ["LEFT", "CENTER", "RIGHT"]
        for i, (occ, name) in enumerate(zip(self.last_occupied, sector_names)):
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            color = (0, 0, 180) if occ else (0, 120, 0)
            cv2.rectangle(img, (x1, 0), (x2, bar_height), color, -1)
            cv2.rectangle(img, (x1, 0), (x2, bar_height), (100, 100, 100), 1)
            status = "BLOCKED" if occ else "FREE"
            cv2.putText(img, f"{name}", (x1 + 10, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img, status, (x1 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # ===== 4. LiDAR 포인트 및 DBSCAN 클러스터 =====
        cluster_colors = [
            (100, 100, 255),  # 빨강 계열
            (100, 255, 100),  # 초록 계열
            (255, 100, 100),  # 파랑 계열
            (100, 255, 255),  # 노랑 계열
            (255, 100, 255),  # 마젠타 계열
            (255, 255, 100),  # 시안 계열
        ]
        
        if len(self.last_points) > 0:
            for i, (x, y) in enumerate(self.last_points):
                # 좌표 변환: X(전방)→위, Y(좌)→좌
                px = int(cx - y * scale)  # Y가 양수면 왼쪽
                py = int(cy - x * scale)  # X가 양수면 위쪽
                
                if 0 <= px < size and 0 <= py < size:
                    if len(self.last_labels) > i:
                        label = self.last_labels[i]
                        if label == -1:
                            color = (80, 80, 80)  # 노이즈: 회색
                        else:
                            color = cluster_colors[label % len(cluster_colors)]
                    else:
                        color = (150, 150, 150)
                    cv2.circle(img, (px, py), 3, color, -1)
        
        # ===== 5. 타겟 장애물 바운딩 박스 =====
        if self.last_target_obs is not None:
            obs = self.last_target_obs
            # 바운딩 박스 좌표 변환
            x1 = int(cx - obs['y_max'] * scale)
            x2 = int(cx - obs['y_min'] * scale)
            y1 = int(cy - obs['x_max'] * scale)
            y2 = int(cy - obs['x_min'] * scale)
            
            # 클램핑
            x1, x2 = max(0, min(x1, x2)), min(size, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(size, max(y1, y2))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"TARGET", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(img, f"dist:{obs['x_min']:.2f}m", (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        # ===== 6. 회피 목표점 (고정된 target) =====
        if self.is_avoiding and self.locked_target_x is not None and self.locked_target_y is not None:
            # 고정된 목표점 좌표 변환
            target_px = int(cx - self.locked_target_y * scale)
            target_py = int(cy - self.locked_target_x * scale)
            
            # 목표점 원 + 화살표
            if 0 <= target_px < size and 0 <= target_py < size:
                cv2.circle(img, (target_px, target_py), 10, (0, 255, 0), -1)  # 녹색 원
                cv2.arrowedLine(img, (cx, cy), (target_px, target_py), 
                              (0, 255, 0), 3, tipLength=0.15)
                cv2.putText(img, f"TARGET ({self.locked_target_x:.2f}, {self.locked_target_y:.2f})", 
                           (target_px - 60, target_py - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # ===== 7. Safe/Stop 거리 표시 =====
        safe_r = int(self.safe_distance * scale)
        stop_r = int(self.stop_distance * scale)
        cv2.circle(img, (cx, cy), safe_r, (0, 180, 180), 1)  # Safe: 노랑
        cv2.circle(img, (cx, cy), stop_r, (0, 0, 255), 1)    # Stop: 빨강
        
        # ===== 8. 로봇 표시 =====
        cv2.circle(img, (cx, cy), 12, (255, 255, 255), -1)
        cv2.arrowedLine(img, (cx, cy), (cx, cy - 35), (255, 255, 255), 2, tipLength=0.3)
        
        # ===== 9. 정보 패널 (하단) =====
        info_y = size - 70
        cv2.rectangle(img, (0, info_y), (size, size), (30, 30, 30), -1)
        
        state_text = self.last_log_state if self.last_log_state else "IDLE"
        if "AVOIDING" in state_text:
            state_color = (0, 255, 255)
        elif "EMERGENCY" in state_text:
            state_color = (0, 0, 255)
        elif "LANE" in state_text or "COMPLETE" in state_text:
            state_color = (0, 255, 0)
        elif "CLEAR" in state_text:
            state_color = (0, 200, 0)
        else:
            state_color = (150, 150, 150)
        
        cv2.putText(img, "DBSCAN OBSTACLE AVOIDANCE", (10, info_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"State: {state_text}", (10, info_y + 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1)
        
        # 회피 상태 정보
        avoid_info = f"Avoiding: {'YES' if self.is_avoiding else 'NO'} | Clear: {self.clear_count}/{self.clear_threshold}"
        cv2.putText(img, avoid_info, (10, info_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(img, f"Points: {len(self.last_points)}", (10, info_y + 68), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # 범례
        cv2.putText(img, "Safe", (size - 100, info_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 180), 1)
        cv2.putText(img, "Stop", (size - 100, info_y + 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(img, "Locked", (size - 100, info_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        try:
            self.pub_debug_image.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(5, f"Debug image error: {e}")


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ObstacleAvoidNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
