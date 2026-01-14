#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco Marker Detector Node (Python OpenCV)
- Detects ArUco markers using OpenCV
- Publishes detected marker IDs and distances
- Publishes visualization image

Topics:
  - /aruco_detector/markers (Int32MultiArray): 감지된 마커 ID 배열
  - /aruco_detector/marker_info (Float32MultiArray): [id, distance, id, distance, ...]
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from cv_bridge import CvBridge


class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector_node')
        
        self.bridge = CvBridge()
        
        # ArUco 딕셔너리 선택 (4x4, 5x5, 6x6 등)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # OpenCV 4.7+ 에서는 ArucoDetector 사용
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.use_new_api = False
        
        # ========================================
        # 카메라 캘리브레이션 파라미터
        # 실제 카메라에 맞게 조정 필요
        # ========================================
        # 카메라 내부 파라미터 (예시값 - LIMO 카메라에 맞게 조정)
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],   # fx, 0, cx
            [0.0, 600.0, 240.0],   # 0, fy, cy
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 왜곡 계수 (k1, k2, p1, p2, k3)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 마커 실제 크기 (미터 단위) - 중요!
        self.marker_size = rospy.get_param('~marker_size', 0.05)  # 5cm 기본값
        
        # Publishers
        self.pub_image = rospy.Publisher('/aruco_detector/image', Image, queue_size=1)
        self.pub_image_compressed = rospy.Publisher('/aruco_detector/image/compressed', CompressedImage, queue_size=1)
        self.pub_markers = rospy.Publisher('/aruco_detector/markers', Int32MultiArray, queue_size=1)
        # 새로운 토픽: [id1, dist1, id2, dist2, ...]
        self.pub_marker_info = rospy.Publisher('/aruco_detector/marker_info', Float32MultiArray, queue_size=1)
        
        # Subscriber
        self.sub_image = rospy.Subscriber(
            '/camera/color/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("ArUco Detector Node initialized")
        rospy.loginfo(f"Dictionary: DICT_4X4_50")
        rospy.loginfo(f"Marker size: {self.marker_size}m")
        rospy.loginfo("Topics:")
        rospy.loginfo("  - /aruco_detector/markers (IDs)")
        rospy.loginfo("  - /aruco_detector/marker_info (ID+Distance)")
        rospy.loginfo("=" * 50)
    
    def detect_markers(self, img):
        """ArUco 마커 감지"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        return corners, ids
    
    def estimate_distance(self, corners):
        """마커와의 거리 추정 (미터)"""
        try:
            # 마커의 3D 좌표 (마커 중심이 원점, Z=0 평면)
            half_size = self.marker_size / 2.0
            obj_points = np.array([
                [-half_size,  half_size, 0],  # top-left
                [ half_size,  half_size, 0],  # top-right
                [ half_size, -half_size, 0],  # bottom-right
                [-half_size, -half_size, 0]   # bottom-left
            ], dtype=np.float32)
            
            distances = []
            rvecs = []
            tvecs = []
            
            for corner in corners:
                # 각 마커에 대해 solvePnP 사용
                img_points = corner[0].astype(np.float32)
                
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    # tvec[2]가 Z축 거리 (카메라로부터의 거리)
                    distance = float(tvec[2][0])
                    distances.append(distance)
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                else:
                    distances.append(0.0)
                    rvecs.append(None)
                    tvecs.append(None)
            
            return distances, rvecs, tvecs
        except Exception as e:
            rospy.logwarn_throttle(5, f"Distance estimation failed: {e}")
            return None, None, None
    
    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            # Decompress image
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return
            
            # Detect markers
            corners, ids = self.detect_markers(img)
            
            # Visualization
            vis_img = img.copy()
            detected_ids = []
            marker_info = []  # [id, distance, id, distance, ...]
            
            if ids is not None and len(corners) > 0:
                # 마커 그리기
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                
                # 거리 추정
                distances, rvecs, tvecs = self.estimate_distance(corners)
                
                for i, marker_id in enumerate(ids.flatten()):
                    detected_ids.append(int(marker_id))
                    
                    # 마커 중심
                    c = corners[i][0]
                    cx = int(np.mean(c[:, 0]))
                    cy = int(np.mean(c[:, 1]))
                    
                    # 거리 정보
                    if distances is not None and i < len(distances):
                        dist = distances[i]
                        marker_info.extend([float(marker_id), dist])
                        
                        # 화면에 ID와 거리 표시
                        cv2.putText(vis_img, f"ID:{marker_id} D:{dist:.2f}m", (cx - 50, cy - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 축 그리기 (선택적)
                        if rvecs is not None and tvecs is not None and rvecs[i] is not None:
                            cv2.drawFrameAxes(vis_img, self.camera_matrix, self.dist_coeffs,
                                            rvecs[i], tvecs[i], self.marker_size * 0.5)
                        
                        rospy.loginfo_throttle(1, f"ArUco ID:{marker_id} Distance:{dist:.3f}m")
                    else:
                        cv2.putText(vis_img, f"ID:{marker_id}", (cx - 30, cy - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 상태 표시
            status_text = f"Markers: {len(detected_ids)}"
            if detected_ids:
                status_text += f" | IDs: {detected_ids}"
            cv2.putText(vis_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Publish markers (기존 호환성)
            if detected_ids:
                marker_msg = Int32MultiArray(data=detected_ids)
                self.pub_markers.publish(marker_msg)
            
            # Publish marker info (ID + Distance)
            if marker_info:
                info_msg = Float32MultiArray(data=marker_info)
                self.pub_marker_info.publish(info_msg)
            
            # Publish image
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            
            # Publish compressed
            _, compressed = cv2.imencode('.jpg', vis_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = rospy.Time.now()
            compressed_msg.format = "jpeg"
            compressed_msg.data = compressed.tobytes()
            self.pub_image_compressed.publish(compressed_msg)
            
        except Exception as e:
            rospy.logerr(f"ArUco detect error: {e}")
    
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ArucoDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
