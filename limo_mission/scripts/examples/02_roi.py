#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 02: ROI (Region of Interest)
- Extracts and visualizes a specific region from the image
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage


# ROI 설정
ROI_X = 320
ROI_Y = 150
ROI_W = 50
ROI_H = 50


# Publisher (전역)
pub_roi = None


def publish_image(publisher, img):
    """이미지를 CompressedImage로 발행"""
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
    publisher.publish(msg)


def image_callback(msg):
    """이미지 콜백"""
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return
    
    # ROI 지정
    roi = img[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    
    # ROI 영역에 사각형 그리기
    cv2.rectangle(roi, (0, 0), (ROI_W-1, ROI_H-1), (0, 255, 0), 2)
    
    # 원본 이미지에도 ROI 위치 표시
    cv2.rectangle(img, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (0, 255, 0), 2)
    cv2.putText(img, "ROI", (ROI_X, ROI_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 발행
    publish_image(pub_roi, img)


def main():
    global pub_roi
    
    rospy.init_node('roi_example')
    
    # Publisher 설정
    pub_roi = rospy.Publisher('/practice/roi/compressed', CompressedImage, queue_size=1)
    
    # Subscriber 설정 (LIMO Pro 카메라 토픽)
    rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, image_callback, queue_size=1)
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("ROI 설정 예제")
    rospy.loginfo(f"ROI: x={ROI_X}, y={ROI_Y}, w={ROI_W}, h={ROI_H}")
    rospy.loginfo("rqt_image_view에서 /practice/roi/compressed 확인")
    rospy.loginfo("=" * 50)
    
    rospy.spin()


if __name__ == '__main__':
    main()
