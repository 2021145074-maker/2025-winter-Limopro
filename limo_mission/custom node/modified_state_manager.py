"""
사용방법:
rosrun limo_mission state_manager.py _initial_state:=LANE_FOLLOWING

이런식으로 원하는 State부터 시작할 수 있다.


런치파일 작성시:
<node pkg="limo_mission" type="state_manager.py" name="state_manager">
    <param name="initial_state" value="LANE_FOLLOWING"/>
</node>

"""


import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String, Bool, Float32
import threading
from cv_bridge import CvBridge


class State:
    # IDLE = "IDLE"
    WAITING_TRAFFIC_LIGHT = "WAITING_TRAFFIC_LIGHT"
    # LANE_FOLLOWING = "LANE_FOLLOWING"
    DETECT_PEDESTRIAN = "DETECT_PEDESTRIAN"
    ROTARY = "ROTARY"
    OBSTACLE_AVOIDANCE = "OBSTACLE_AVOIDANCE"
    PARKING = "PARKING"

class StateManager:
    def __init__(self):
        rospy.init_node('state_manager2')
        
        # Get initial state from parameter (default: IDLE)
        initial_state_param = rospy.get_param('~initial_state', 'WAITING_TRAFFIC_LIGHT')
        
        # State transition order
        self.state_order = [
            # State.IDLE,
            State.WAITING_TRAFFIC_LIGHT,
            # State.LANE_FOLLOWING,
            State.DETECT_PEDESTRIAN,
            State.ROTARY,
            State.OBSTACLE_AVOIDANCE,
            State.PARKING
        ]

        self.current_state = "WAITING_TRAFFIC_LIGHT"
        
        # Validate and set initial state
        if hasattr(State, initial_state_param) and getattr(State, initial_state_param) in self.state_order:
            self.current_state = getattr(State, initial_state_param)
            rospy.loginfo(f"Starting with state: {self.current_state}")
        else:
            self.current_state = State.WAITING_TRAFFIC_LIGHT
            rospy.logwarn(f"Invalid initial state '{initial_state_param}', defaulting to IDLE")
        
        # HLS range for yellow - more robust to lighting
        self.YELLOW_LOWER_HLS = np.array([15, 50, 200])   # [H, L, S]
        self.YELLOW_UPPER_HLS = np.array([35, 255, 255]) # [H, L, S]

        self.width = 640
        self.height = 480

        self.ROI_TOP_LEFT = (0, int(self.height*0.7))
        self.ROI_BOTTOM_RIGHT = (self.width, self.height)

        # Yellow line detection parameters
        self.yellow_detected_count = 0  # 연속 감지 카운터
        self.required_detections = 5    # 필요한 연속 감지 횟수
        self.yellow_threshold = 2000    # ROI 내 노란색 픽셀 수 임계값 (조정 가능)
        
        # State change lock
        self.last_state_change_time = rospy.Time.now()
        self.state_lock_duration = rospy.Duration(10.0)  # 10초 잠금
        
        # Pedestrian state speed control
        self.pedestrian_slow_speed = 0.3   # 처음 N초 동안의 느린 속도
        self.pedestrian_normal_speed = 0.8  # 이후 정상 속도
        self.pedestrian_slow_duration = 10.0  # 느린 속도 유지 시간 (초)
        self.speed_timer = None

        # Traffic light state
        self.current_traffic_state = False  # False: Green, True: Red
        self.previous_traffic_state = False  # 이전 신호등 상태
        self.traffic_light_recognized = False  # 신호등이 한 번이라도 인식되었는지
        self.green_light_time = None  # 초록불로 바뀐 시간
        self.green_light_grace_period = rospy.Duration(3.0)  # 초록불 후 3초간 노란선 감지 무시

        # === 미션별 Base Speed 설정 ===
        self.mission_speeds = {
            State.WAITING_TRAFFIC_LIGHT: 0.8,  # 신호등 대기 후 출발
            State.DETECT_PEDESTRIAN: 1.0,      # 횡단보도 - 느리게
            State.ROTARY: 0.5,                 # 로터리
            State.OBSTACLE_AVOIDANCE: 0.7,   # 장애물 회피
            State.PARKING: 0.5,               # 주차 - 가장 느리게
        }
        self.default_speed = 1.0  # 기본 속도

        self.bridge = CvBridge()

        # Subscribers and Publishers
        self.sub_image = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1)
        self.sub_traffic_state = rospy.Subscriber('/limo/traffic_light/state', Bool, self.traffic_light_callback, queue_size=1)
        self.sub_lane_following_state = rospy.Subscriber('/state_manager/lane_state', String, self.lane_state_callback, queue_size=1)
        # LANE_FOLLOWING -> Go

        self.pub_state = rospy.Publisher('/state_manager/state', String, queue_size=1)
        self.pub_debug_image = rospy.Publisher('/state_manager/debug_image/compressed', CompressedImage, queue_size=1)
        self.pub_base_speed = rospy.Publisher('/lane_detection/base_speed', Float32, queue_size=1)

        # Publish initial state after a short delay to ensure subscribers are ready
        rospy.sleep(0.5)
        self.pub_state.publish(self.current_state)
        rospy.loginfo(f"Published initial state: {self.current_state}")


    def image_callback(self, msg: CompressedImage):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.detect_yellow_line(cv_image)

    def traffic_light_callback(self, msg: Bool):
        # 신호등 인식됨 표시
        if not self.traffic_light_recognized:
            self.traffic_light_recognized = True
            rospy.loginfo("Traffic light RECOGNIZED for the first time")
        
        # 빨간불(True) -> 초록불(False)로 바뀌는 순간 감지
        if self.previous_traffic_state and not msg.data:
            self.green_light_time = rospy.Time.now()
            self.yellow_detected_count = 0  # 카운터 리셋
            rospy.loginfo("Traffic light changed to GREEN, starting grace period")
        
        self.previous_traffic_state = self.current_traffic_state
        self.current_traffic_state = msg.data

    def lane_state_callback(self, msg: String):
        self.lane_following_state = msg.data

    def detect_yellow_line(self, image):
        # Create debug image - darken the original
        debug_image = (image * 0.5).astype(np.uint8)
        
        # PARKING 상태에서는 노란선 감지 무시 (마지막 상태)
        is_parking = (self.current_state == State.PARKING)
        
        # Check if state is locked
        is_locked = rospy.Time.now() - self.last_state_change_time < self.state_lock_duration
        
        # Check if in green light grace period (초록불 후 3초간 노란선 감지 무시)
        in_grace_period = False
        if self.green_light_time is not None:
            if rospy.Time.now() - self.green_light_time < self.green_light_grace_period:
                in_grace_period = True
            else:
                self.green_light_time = None  # Grace period 종료
        
        # WAITING_TRAFFIC_LIGHT 상태에서 신호등 미인식 또는 빨간불이면 노란선 감지 무시
        waiting_for_traffic = False
        if self.current_state == State.WAITING_TRAFFIC_LIGHT:
            if not self.traffic_light_recognized:
                waiting_for_traffic = True  # 신호등 아직 인식 안됨
                rospy.loginfo_throttle(2.0, "Waiting for traffic light to be recognized...")
            elif self.current_traffic_state:  # 빨간불
                waiting_for_traffic = True
                rospy.loginfo_throttle(2.0, "Waiting for GREEN light...")
        
        # Step 1: Extract ROI first
        roi_image = image[self.ROI_TOP_LEFT[1]:self.ROI_BOTTOM_RIGHT[1], 
                          self.ROI_TOP_LEFT[0]:self.ROI_BOTTOM_RIGHT[0]]
        
        # Step 2: Convert ROI to HLS and apply yellow mask
        roi_hls = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HLS)
        roi_mask = cv2.inRange(roi_hls, self.YELLOW_LOWER_HLS, self.YELLOW_UPPER_HLS)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to reduce small noise
        roi_mask = cv2.GaussianBlur(roi_mask, (5, 5), 0)
        _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create full-size mask for visualization
        full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        full_mask[self.ROI_TOP_LEFT[1]:self.ROI_BOTTOM_RIGHT[1], 
                  self.ROI_TOP_LEFT[0]:self.ROI_BOTTOM_RIGHT[0]] = roi_mask
        
        # Overlay yellow detection on debug image
        yellow_overlay = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        yellow_overlay[:, :, 0] = 0  # Remove blue channel
        yellow_overlay[:, :, 1] = full_mask  # Green channel
        yellow_overlay[:, :, 2] = full_mask  # Red channel (makes it yellow)
        debug_image = cv2.addWeighted(debug_image, 1.0, yellow_overlay, 0.7, 0)
        
        # Draw ROI rectangle
        cv2.rectangle(debug_image, self.ROI_TOP_LEFT, self.ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
        
        # Count yellow pixels in ROI
        yellow_pixel_count = cv2.countNonZero(roi_mask)
        
        if not is_locked and not in_grace_period and not is_parking and not waiting_for_traffic:
            if yellow_pixel_count > self.yellow_threshold:
                self.yellow_detected_count += 1
                rospy.loginfo(f"Yellow line detected: {self.yellow_detected_count}/{self.required_detections}")
                
                # Change state after required consecutive detections
                if self.yellow_detected_count >= self.required_detections:
                    self.change_to_next_state()
                    self.yellow_detected_count = 0  # Reset counter
            else:
                self.yellow_detected_count = 0  # Reset if not detected
        elif in_grace_period or is_parking or waiting_for_traffic:
            self.yellow_detected_count = 0  # Grace period, PARKING, 또는 신호등 대기 중 카운터 리셋 유지
        
        # Add state text to top-left corner
        state_text = f"State: {self.current_state}"
        count_text = f"Count: {self.yellow_detected_count}/{self.required_detections}"
        lock_text = f"Locked: {is_locked}"
        grace_text = f"Grace Period: {in_grace_period}"
        parking_text = f"Parking (skip): {is_parking}"
        pixel_text = f"Yellow pixels: {yellow_pixel_count}"
        traffic_text = f"Traffic: {'RED' if self.current_traffic_state else 'GREEN'} (recognized: {self.traffic_light_recognized})"
        waiting_text = f"Waiting for traffic: {waiting_for_traffic}"
        
        cv2.putText(debug_image, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_image, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, lock_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, grace_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, parking_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, pixel_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, traffic_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not self.current_traffic_state else (0, 0, 255), 2)
        cv2.putText(debug_image, waiting_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Publish debug image
        debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_image, dst_format='jpeg')
        self.pub_debug_image.publish(debug_msg)
    
    def change_to_next_state(self):
        try:
            # Special condition: if current state is WAITING_TRAFFIC_LIGHT, check traffic state
            if self.current_state == State.WAITING_TRAFFIC_LIGHT:
                if self.current_traffic_state:  # True means RED light
                    rospy.logwarn("Cannot transition from WAITING_TRAFFIC_LIGHT: traffic light is RED")
                    self.yellow_detected_count = 0  # 빨간불일 때 카운터 리셋
                    return
                else:
                    rospy.loginfo("Traffic light is GREEN, transitioning to next state")
            
            current_index = self.state_order.index(self.current_state)
            next_index = (current_index + 1) % len(self.state_order)
            self.current_state = self.state_order[next_index]
            
            # Lock state changes (PEDESTRIAN은 20초, 나머지는 10초)
            if self.current_state == State.DETECT_PEDESTRIAN:
                self.state_lock_duration = rospy.Duration(20.0)
            else:
                self.state_lock_duration = rospy.Duration(10.0)
            self.last_state_change_time = rospy.Time.now()
            
            # Publish new state
            self.pub_state.publish(self.current_state)
            rospy.loginfo(f"State changed to: {self.current_state}")
            
            # === 미션별 속도 설정 발행 ===
            self._publish_mission_speed(self.current_state)
                
        except ValueError:
            rospy.logerr(f"Current state {self.current_state} not in state order")
    
    def _publish_mission_speed(self, state):
        """미션별 속도 설정 발행"""
        # 기존 타이머 취소
        if self.speed_timer is not None:
            self.speed_timer.cancel()
            self.speed_timer = None
        
        # PEDESTRIAN 상태일 경우 처음에는 느린 속도로
        if state == State.DETECT_PEDESTRIAN:
            self.pub_base_speed.publish(Float32(data=self.pedestrian_slow_speed))
            rospy.loginfo(f"[{state}] Base speed set to SLOW: {self.pedestrian_slow_speed}")
            
            # 일정 시간 후 정상 속도로 복귀
            self.speed_timer = threading.Timer(
                self.pedestrian_slow_duration, 
                self._restore_normal_speed
            )
            self.speed_timer.start()
        else:
            # 다른 미션은 mission_speeds 테이블 사용
            speed = self.mission_speeds.get(state, self.default_speed)
            self.pub_base_speed.publish(Float32(data=speed))
            rospy.loginfo(f"[{state}] Base speed set to: {speed}")
    
    def _restore_normal_speed(self):
        """정상 속도로 복귀 (PEDESTRIAN 이후)"""
        self.pub_base_speed.publish(Float32(data=self.default_speed))
        rospy.loginfo(f"[PEDESTRIAN] Speed restored to default: {self.default_speed}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        manager = StateManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass