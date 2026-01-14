#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 03: LiDAR Distance Control
- Measures front distance using LiDAR
- Controls speed based on obstacle distance
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


TARGET_DISTANCE = 0.3   # 목표 거리 (m)
STOP_DISTANCE = 0.15    # 정지 거리 (m)
NORMAL_SPEED = 0.5      # 정상 속도 (m/s)
SLOW_SPEED = 0.2        # 감속 속도 (m/s)
FRONT_WINDOW = 50       # 전방 감지 범위 (포인트 수)


# Publisher (전역)
drive_pub = None


def scan_callback(msg):
    """LiDAR 콜백"""
    ranges = np.array(msg.ranges)
    total = len(ranges)
    
    # LIMO Pro YDLidar: -180도 ~ +180도 스캔
    # 인덱스 중간(center)이 전방(0도)
    center = total // 2
    half_window = FRONT_WINDOW // 2
    front_start = max(0, center - half_window)
    front_end = min(total, center + half_window)
    front_ranges = ranges[front_start:front_end]
    
    # 최소 거리 계산 (0이나 무한대 제외)
    valid = front_ranges[(front_ranges > 0.01) & (front_ranges < 10.0)]
    
    if len(valid) == 0:
        min_dist = 10.0
    else:
        min_dist = np.min(valid)
    
    # 속도 결정
    if min_dist < STOP_DISTANCE:
        speed = 0.0
        status = "STOP"
    elif min_dist < TARGET_DISTANCE:
        speed = SLOW_SPEED
        status = "SLOW"
    else:
        speed = NORMAL_SPEED
        status = "GO"
    
    # 명령 발행 (LIMO Pro는 Twist 사용)
    cmd = Twist()
    cmd.linear.x = speed
    cmd.angular.z = 0.0
    drive_pub.publish(cmd)
    
    # 터미널 출력
    rospy.loginfo_throttle(0.3, f"전방: {min_dist:.2f}m | {status} | {speed:.1f}m/s")


def main():
    global drive_pub
    
    rospy.init_node('lidar_distance')
    
    # Publisher 설정 (LIMO Pro 모터 명령 토픽)
    drive_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    
    # Subscriber 설정 (LIMO Pro LiDAR 토픽)
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("LiDAR 거리 유지 시작!")
    rospy.loginfo(f"목표 거리: {TARGET_DISTANCE}m, 정지 거리: {STOP_DISTANCE}m")
    rospy.loginfo("=" * 50)
    
    rospy.spin()


if __name__ == '__main__':
    main()
