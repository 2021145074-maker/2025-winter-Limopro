#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parking Mission Node for LIMO Pro
- 후진 평행주차 수행
- ArUco Marker ID 0이 일정 거리 이내일 때 자동 시작
- 시간 기반 제어
- 주차 완료 후 정지 (라인트레이싱 종료)
"""

import rospy
from std_msgs.msg import Bool, String, Float32MultiArray
from geometry_msgs.msg import Twist


class ParkingState:
    IDLE = "IDLE"                      # 대기 (라인트레이싱 중)
    # === 주차 ===
    STOP_BEFORE = "STOP_BEFORE"        # 주차 구역 앞에서 정지
    FORWARD_PASS = "FORWARD_PASS"      # 주차 구역 지나서 전진
    STOP_ALIGN = "STOP_ALIGN"          # 정렬 위해 정지
    REVERSE_RIGHT = "REVERSE_RIGHT"    # 후진 + 우회전
    REVERSE_LEFT = "REVERSE_LEFT"      # 후진 + 좌회전 (정렬)
    FORWARD_STRAIGHT = "FORWARD_STRAIGHT"  # 전진 직진 (진입)
    DONE = "DONE"                      # 주차 완료 (종료)


class ParkingNode:
    def __init__(self):
        rospy.init_node('parking_node')
        
        # ========================================
        # 파라미터 (yaml에서 로드 또는 기본값 사용)
        # ========================================
        self.speed_slow = rospy.get_param('~speed_slow', 0.15)           # 저속
        self.speed_reverse = rospy.get_param('~speed_reverse', -0.13)    # 후진 속도
        self.steering_angle = rospy.get_param('~steering_angle', 0.8)    # 조향 각도
        
        self.forward_time = rospy.get_param('~forward_time', 1.5)              # 주차 구역 지나 전진 시간 (초)
        self.reverse_right_time = rospy.get_param('~reverse_right_time', 3.0)  # 후진 우회전 시간 (초)
        self.reverse_left_time = rospy.get_param('~reverse_left_time', 3.0)    # 후진 좌회전 시간 (초)
        self.final_forward_time = rospy.get_param('~final_forward_time', 1.0)  # 마지막 전진 직진 시간 (초)
        
        # 트리거 거리 (미터) - 이 거리 이내일 때 주차 시작
        self.trigger_distance = rospy.get_param('~trigger_distance', 0.5)  # 0.5m
        self.target_marker_id = rospy.get_param('~target_marker_id', 0)    # 마커 ID 0
        
        # ========================================
        # 상태
        # ========================================
        self.state = ParkingState.IDLE
        self.phase_start_time = None
        self.parking_triggered = False  # 주차 트리거 (1회만)
        
        # ========================================
        # ROS
        # ========================================
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_state = rospy.Publisher('/parking/state', String, queue_size=1)
        self.pub_done = rospy.Publisher('/parking/done', Bool, queue_size=1)
        
        # ArUco 마커 정보 구독 (ID + 거리)
        self.sub_aruco = rospy.Subscriber('/aruco_detector/marker_info', Float32MultiArray, self.aruco_callback, queue_size=1)
        
        # 제어 루프 (20Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Parking Node initialized")
        rospy.loginfo(f"Trigger: ArUco Marker ID {self.target_marker_id}")
        rospy.loginfo(f"Trigger distance: {self.trigger_distance}m")
        rospy.loginfo("=" * 50)
    
    def aruco_callback(self, msg):
        """ArUco 마커 정보 콜백 - ID가 target이고 거리가 임계값 이내면 주차 시작
        
        msg.data 형식: [id1, distance1, id2, distance2, ...]
        """
        if self.state != ParkingState.IDLE or self.parking_triggered:
            return
        
        # 데이터 파싱: [id, dist, id, dist, ...]
        data = msg.data
        if len(data) < 2:
            return
        
        for i in range(0, len(data) - 1, 2):
            marker_id = int(data[i])
            distance = data[i + 1]
            
            if marker_id == self.target_marker_id:
                rospy.loginfo_throttle(0.5, f"ArUco ID:{marker_id} Distance:{distance:.3f}m (trigger: {self.trigger_distance}m)")
                
                if distance <= self.trigger_distance:
                    rospy.loginfo(f"ArUco Marker ID {marker_id} within {self.trigger_distance}m! Starting parking...")
                    self.parking_triggered = True
                    self.state = ParkingState.STOP_BEFORE
                    self.phase_start_time = rospy.Time.now()
                    return
    
    def elapsed_time(self):
        """현재 페이즈 경과 시간"""
        if self.phase_start_time is None:
            return 0
        return (rospy.Time.now() - self.phase_start_time).to_sec()
    
    def next_phase(self, new_state):
        """다음 페이즈로 전환"""
        rospy.loginfo(f"Phase: {self.state} -> {new_state}")
        self.state = new_state
        self.phase_start_time = rospy.Time.now()
    
    def control_loop(self, event):
        """메인 제어 루프"""
        cmd = Twist()
        
        # 상태 발행
        self.pub_state.publish(String(data=self.state))
        
        # ========================================
        # IDLE: 대기
        # ========================================
        if self.state == ParkingState.IDLE:
            cmd.linear.x = 0
            cmd.angular.z = 0
        
        # ========================================
        # STOP_BEFORE: 잠시 정지 (1초)
        # ========================================
        elif self.state == ParkingState.STOP_BEFORE:
            cmd.linear.x = 0
            cmd.angular.z = 0
            
            if self.elapsed_time() > 1.0:
                self.next_phase(ParkingState.FORWARD_PASS)
        
        # ========================================
        # FORWARD_PASS: 주차 구역 지나서 전진
        # ========================================
        elif self.state == ParkingState.FORWARD_PASS:
            cmd.linear.x = self.speed_slow
            cmd.angular.z = 0
            
            if self.elapsed_time() > self.forward_time:
                self.next_phase(ParkingState.STOP_ALIGN)
        
        # ========================================
        # STOP_ALIGN: 정렬 위해 정지 (1초)
        # ========================================
        elif self.state == ParkingState.STOP_ALIGN:
            cmd.linear.x = 0
            cmd.angular.z = 0
            
            if self.elapsed_time() > 1.0:
                self.next_phase(ParkingState.REVERSE_RIGHT)
        
        # ========================================
        # REVERSE_RIGHT: 후진 + 우회전 (시간 기반)
        # ========================================
        elif self.state == ParkingState.REVERSE_RIGHT:
            cmd.linear.x = self.speed_reverse
            cmd.angular.z = -self.steering_angle  # 우회전
            
            rospy.loginfo_throttle(0.5, f"REVERSE_RIGHT: {self.elapsed_time():.1f}s / {self.reverse_right_time}s")
            
            if self.elapsed_time() >= self.reverse_right_time:
                self.next_phase(ParkingState.REVERSE_LEFT)
        
        # ========================================
        # REVERSE_LEFT: 후진 + 좌회전 (시간 기반)
        # ========================================
        elif self.state == ParkingState.REVERSE_LEFT:
            cmd.linear.x = self.speed_reverse
            cmd.angular.z = self.steering_angle  # 좌회전
            
            rospy.loginfo_throttle(0.5, f"REVERSE_LEFT: {self.elapsed_time():.1f}s / {self.reverse_left_time}s")
            
            if self.elapsed_time() >= self.reverse_left_time:
                self.next_phase(ParkingState.FORWARD_STRAIGHT)
        
        # ========================================
        # FORWARD_STRAIGHT: 전진 직진 (주차 구역 안으로)
        # ========================================
        elif self.state == ParkingState.FORWARD_STRAIGHT:
            cmd.linear.x = self.speed_slow
            cmd.angular.z = 0
            
            if self.elapsed_time() > self.final_forward_time:
                self.next_phase(ParkingState.DONE)
        
        # ========================================
        # DONE: 주차 완료 - 정지 (미션 종료)
        # ========================================
        elif self.state == ParkingState.DONE:
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.pub_done.publish(Bool(data=True))
            rospy.loginfo_throttle(3, "Parking complete! Mission finished.")
        
        # cmd_vel 발행
        self.pub_cmd.publish(cmd)
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ParkingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
