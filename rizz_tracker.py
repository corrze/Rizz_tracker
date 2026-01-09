import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time
import os
import urllib.request
import urllib.error
import mediapipe as mp

# MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class RizzTracker:
    def __init__(self, camera_index=0):
        """
        Initialize the Rizz Tracker with pose detection and fashion detection.
        Uses MediaPipe Tasks API (0.10.30+)
        
        Args:
            camera_index: Index of the webcam to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        
        # Initialize MediaPipe Pose using Tasks API
        try:
            # Download model file if it doesn't exist
            model_path = self._download_pose_model()
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            print("✓ MediaPipe Pose Landmarker initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            raise
        
        # Initialize YOLO for clothing/fashion detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # This will download on first run
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"Warning: YOLO model loading failed: {e}")
            self.yolo_model = None
        
        # Rizz scoring parameters
        self.pose_history = deque(maxlen=30)  # Track last 30 frames
        self.clothing_detections = deque(maxlen=60)  # Track clothing over time
        self.rizz_score = 0.0
        self.rizz_history = deque(maxlen=10)
        # Last sub-scores (for tips / debug)
        self.last_movement_score = 0.0
        self.last_posture_score = 0.0
        self.last_style_score = 0.0
        self.last_fashion_score = 0.0
        
        # Pose landmark indices (MediaPipe Pose has 33 landmarks)
        # Mapping to match the old API structure
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.NOSE = 0
        self.LEFT_EYE = 2
        self.RIGHT_EYE = 5
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8
        
        # Pose connections for drawing skeleton (pairs of landmark indices)
        self.POSE_CONNECTIONS = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),    # Face outline
            (0, 4), (4, 5), (5, 6), (6, 8),    # Face outline
            # Upper body
            (9, 10),                            # Head to neck
            (11, 12),                           # Shoulders
            (11, 13), (13, 15),                 # Left arm
            (12, 14), (14, 16),                 # Right arm
            (11, 23), (12, 24),                 # Torso
            (23, 24),                           # Hips
            # Lower body
            (23, 25), (25, 27),                 # Left leg
            (24, 26), (26, 28),                 # Right leg
        ]
        
        # Visualization toggle
        self.show_pose_visualization = False
        # Tips box toggle
        self.show_tips_box = True
        
        # Movement thresholds
        self.movement_threshold = 0.02
    
    def _download_pose_model(self):
        """
        Download the MediaPipe pose landmarker model if it doesn't exist.
        
        Returns:
            Path to the model file
        """
        # Try lite first (smallest, fastest), then full if lite fails
        model_variants = [
            ("pose_landmarker_lite.task", 
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
            ("pose_landmarker_full.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
        ]
        
        for model_filename, model_url in model_variants:
            model_path = os.path.join(os.path.dirname(__file__), model_filename)
            
            if os.path.exists(model_path):
                print(f"✓ Using existing model: {model_filename}")
                return model_path
            
            # Try to download
            print(f"Downloading {model_filename}...")
            print(f"URL: {model_url}")
            
            try:
                # Show progress
                def show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100.0 / total_size) if total_size > 0 else 0
                    print(f"\rDownload progress: {percent:.1f}%", end='', flush=True)
                
                urllib.request.urlretrieve(model_url, model_path, show_progress)
                print(f"\n✓ Model downloaded successfully: {model_path}")
                return model_path
            except urllib.error.HTTPError as e:
                print(f"\n✗ Failed to download {model_filename}: HTTP {e.code}")
                if model_filename != model_variants[-1][0]:  # If not the last variant, try next
                    print(f"Trying alternative model...")
                    continue
                raise
            except Exception as e:
                print(f"\n✗ Error downloading {model_filename}: {e}")
                if model_filename != model_variants[-1][0]:  # If not the last variant, try next
                    print(f"Trying alternative model...")
                    continue
                raise
        
        # If all downloads failed
        raise FileNotFoundError(
            "Could not download pose landmarker model. Please manually download from:\n"
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task\n"
            f"And save it as: {os.path.join(os.path.dirname(__file__), 'pose_landmarker_full.task')}"
        )
        
    def get_landmark(self, landmarks, index):
        """Helper to get landmark from Tasks API format"""
        if not landmarks or len(landmarks) == 0:
            return None
        if index >= len(landmarks):
            return None
        return landmarks[index]
    
    def calculate_movement_score(self, current_landmarks):
        """
        Calculate movement score based on pose changes over time.
        More strict scoring: low movement = low score, high movement = high score.
        
        Args:
            current_landmarks: Current pose landmarks (list from Tasks API)
            
        Returns:
            Movement score (0-100)
        """
        if not current_landmarks or len(current_landmarks) == 0:
            return 20.0  # Low baseline for no movement
            
        if len(self.pose_history) < 5:
            self.pose_history.append(current_landmarks)
            return 20.0  # Low baseline when starting
        
        # Calculate average movement of key points
        previous_landmarks = self.pose_history[-5]  # Compare to 5 frames ago
        total_movement = 0.0
        point_count = 0
        
        key_point_indices = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_ANKLE, self.RIGHT_ANKLE,
            self.LEFT_WRIST, self.RIGHT_WRIST,
        ]
        
        for idx in key_point_indices:
            curr_landmark = self.get_landmark(current_landmarks, idx)
            prev_landmark = self.get_landmark(previous_landmarks, idx)
            
            if curr_landmark and prev_landmark:
                # Tasks API format: landmark has x, y, z, visibility
                if (hasattr(curr_landmark, 'visibility') and hasattr(prev_landmark, 'visibility') and
                    curr_landmark.visibility > 0.5 and prev_landmark.visibility > 0.5):
                    
                    curr_x = curr_landmark.x
                    curr_y = curr_landmark.y
                    prev_x = prev_landmark.x
                    prev_y = prev_landmark.y
                    
                    movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    total_movement += movement
                    point_count += 1
        
        if point_count == 0:
            movement_score = 20.0  # Low baseline
        else:
            avg_movement = total_movement / point_count
            # More strict scaling: need significant movement to get high scores
            # Very small movements (< 0.005) = 20-40 score
            # Medium movements (0.005-0.02) = 40-70 score  
            # Large movements (> 0.02) = 70-100 score
            if avg_movement < 0.002:
                movement_score = 20.0  # Almost no movement
            elif avg_movement < 0.005:
                movement_score = 20 + (avg_movement / 0.005) * 20  # 20-40
            elif avg_movement < 0.02:
                movement_score = 40 + ((avg_movement - 0.005) / 0.015) * 30  # 40-70
            else:
                movement_score = min(100, 70 + ((avg_movement - 0.02) / 0.03) * 30)  # 70-100
        
        self.pose_history.append(current_landmarks)
        return movement_score
    
    def calculate_posture_score(self, landmarks):
        """
        Calculate posture/confidence score based on pose.
        More strict: requires good posture to get high scores.
        
        Args:
            landmarks: Pose landmarks (list from Tasks API)
            
        Returns:
            Posture score (0-100)
        """
        if not landmarks or len(landmarks) == 0:
            return 30.0  # Lower baseline
        
        left_shoulder = self.get_landmark(landmarks, self.LEFT_SHOULDER)
        right_shoulder = self.get_landmark(landmarks, self.RIGHT_SHOULDER)
        nose = self.get_landmark(landmarks, self.NOSE)
        
        if not left_shoulder or not right_shoulder or not nose:
            return 30.0  # Lower baseline
        
        if (hasattr(left_shoulder, 'visibility') and hasattr(right_shoulder, 'visibility') and
            hasattr(nose, 'visibility')):
            if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
                nose.visibility < 0.5):
                return 30.0  # Lower baseline
        
        # Calculate shoulder levelness (more strict)
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        # Perfect level (diff < 0.01) = 100, slight tilt (0.01-0.02) = 80-100, bad (>0.02) = 0-80
        if shoulder_diff < 0.01:
            shoulder_level_score = 100
        elif shoulder_diff < 0.02:
            shoulder_level_score = 100 - ((shoulder_diff - 0.01) / 0.01) * 20  # 80-100
        else:
            shoulder_level_score = max(0, 80 - ((shoulder_diff - 0.02) / 0.03) * 80)  # 0-80
        
        # Check if head is above shoulders (good posture) - more strict
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        head_shoulder_diff = avg_shoulder_y - nose.y  # Positive if head is above shoulders
        if head_shoulder_diff > 0.15:  # Head well above shoulders
            head_position_score = 100
        elif head_shoulder_diff > 0.10:  # Head moderately above
            head_position_score = 70 + ((head_shoulder_diff - 0.10) / 0.05) * 30  # 70-100
        elif head_shoulder_diff > 0.05:  # Head slightly above
            head_position_score = 40 + ((head_shoulder_diff - 0.05) / 0.05) * 30  # 40-70
        else:  # Head at or below shoulders (bad posture)
            head_position_score = max(0, 40 * (head_shoulder_diff / 0.05))  # 0-40
        
        # Combine scores (weighted average)
        posture_score = (shoulder_level_score * 0.6 + head_position_score * 0.4)
        
        return posture_score
    
    def calculate_pose_style_score(self, landmarks):
        """
        Pose / face style score (0-100).
        Focuses on head alignment, centering, and stillness (holding a pose),
        not big body movement.
        """
        if not landmarks or len(landmarks) == 0:
            return 40.0

        # Use head landmarks
        nose = self.get_landmark(landmarks, self.NOSE)
        left_eye = self.get_landmark(landmarks, self.LEFT_EYE)
        right_eye = self.get_landmark(landmarks, self.RIGHT_EYE)

        if not nose or not left_eye or not right_eye:
            return 40.0

        # 1) Head tilt (based on eye line angle)
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        if abs(dx) < 1e-6:
            dx = 1e-6
        angle_rad = np.arctan2(dy, dx)
        angle_deg = abs(angle_rad * 180.0 / np.pi)

        # Small tilt (0–10 deg) = best, 10–25 deg = okay, >25 deg = worse
        if angle_deg <= 10:
            tilt_score = 100.0
        elif angle_deg <= 25:
            tilt_score = 100.0 - ((angle_deg - 10) / 15.0) * 40.0  # 60–100
        else:
            extra = min(35.0, angle_deg - 25.0)
            tilt_score = max(0.0, 60.0 - (extra / 35.0) * 60.0)    # 0–60

        # 2) Face centering (nose near center horizontally)
        offset_from_center = abs(nose.x - 0.5)
        if offset_from_center <= 0.15:
            center_score = 100.0 - (offset_from_center / 0.15) * 30.0  # 70–100
        elif offset_from_center <= 0.30:
            center_score = 70.0 - ((offset_from_center - 0.15) / 0.15) * 30.0  # 40–70
        else:
            center_score = 40.0

        # 3) Head stillness (holding a pose)
        pose_history = list(self.pose_history)
        if len(pose_history) >= 5:
            old_landmarks = pose_history[-5]
            old_nose = self.get_landmark(old_landmarks, self.NOSE) if old_landmarks else None
        else:
            old_nose = None

        if old_nose:
            dist = np.sqrt((nose.x - old_nose.x) ** 2 + (nose.y - old_nose.y) ** 2)
            if dist < 0.002:
                stillness_score = 100.0
            elif dist < 0.008:
                stillness_score = 70.0 + (1 - (dist - 0.002) / 0.006) * 30.0  # 70–100
            elif dist < 0.03:
                stillness_score = 30.0 + (1 - (dist - 0.008) / 0.022) * 40.0  # 30–70
            else:
                stillness_score = 0.0
        else:
            stillness_score = 60.0

        style_score = (
            tilt_score * 0.4 +
            center_score * 0.3 +
            stillness_score * 0.3
        )

        return style_score

    def calculate_fashion_score(self, detections):
        """
        Calculate fashion score based on detected clothing/accessories.
        More strict: requires actual accessories to get high scores.
        
        Args:
            detections: YOLO detection results
            
        Returns:
            Fashion score (0-100)
        """
        if not detections or len(detections) == 0:
            return 25.0  # Lower baseline - no detections
        
        # Store detections
        detected_items = []
        person_count = 0
        
        for detection in detections:
            boxes = detection.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get class name (requires class names, using simple logic)
                    if class_id == 0:  # COCO class 0 is 'person'
                        person_count += 1
                    elif confidence > 0.5:  # Higher threshold for accessories
                        detected_items.append((class_id, confidence))
        
        # Base score on number of detected items and person detection
        if person_count == 0:
            fashion_score = 0.0
        else:
            # Person detected = base 25 points (not 50)
            # Each accessory adds points, but need multiple to get high scores
            base_score = 25
            item_bonus = min(75, len(detected_items) * 15)  # Each item worth 15 points, max 75
            fashion_score = base_score + item_bonus
        
        self.clothing_detections.append(fashion_score)
        
        # Add bonus for fashion changes (new items detected) - but smaller bonus
        if len(self.clothing_detections) > 5:
            recent_avg = np.mean(list(self.clothing_detections)[-5:])
            older_avg = np.mean(list(self.clothing_detections)[-10:-5]) if len(self.clothing_detections) > 10 else recent_avg
            change_bonus = abs(recent_avg - older_avg) * 1.5  # Smaller bonus multiplier
            fashion_score = min(100, fashion_score + change_bonus)
        
        return fashion_score
    
    def get_rizz_tips(self, movement_score, posture_score, style_score, fashion_score):
        """
        Generate short tips on how to raise the rizz meter based on sub-scores.
        """
        tips = []

        # Style / pose tips
        if style_score < 60:
            tips.append("Hold a steady pose facing the camera")
        if posture_score < 60:
            tips.append("Stand taller with level shoulders")

        # Movement tips (small influence)
        if movement_score < 40:
            tips.append("Add a bit of smooth motion")
        elif movement_score > 80:
            tips.append("Move less erratically")

        # Fashion tips
        if fashion_score < 40:
            tips.append("Show an accessory (hat, bag, etc.)")

        if not tips:
            tips.append("Maintain your pose, you're cooking")

        return tips[:3]

    def draw_rizz_tips_box(self, frame, tips):
        """
        Draw a small box with tips on how to raise rizz.
        """
        if not tips or not self.show_tips_box:
            return

        h, w = frame.shape[:2]
        x, y = 10, 60  # top-left corner of box
        padding = 8
        line_height = 20

        lines = ["Tips to raise rizz:"] + [f"- {t}" for t in tips]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        max_width = 0
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, tw)

        box_height = padding * 2 + line_height * len(lines)
        box_width = padding * 2 + max_width

        # Background
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (255, 255, 255), 1)

        # Text
        ty = y + padding + 15
        for i, line in enumerate(lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (x + padding, ty + i * line_height),
                        font, font_scale, color, thickness, cv2.LINE_AA)

    def calculate_rizz_score(self, landmarks, detections):
        """
        Calculate overall rizz score combining movement, posture, and fashion.
        
        Args:
            landmarks: Pose landmarks
            detections: Fashion/clothing detections
            
        Returns:
            Rizz score (0-100)
        """
        movement_score = self.calculate_movement_score(landmarks)
        posture_score = self.calculate_posture_score(landmarks)
        style_score = self.calculate_pose_style_score(landmarks)
        fashion_score = self.calculate_fashion_score(detections)

        # Store for tips/debug
        self.last_movement_score = movement_score
        self.last_posture_score = posture_score
        self.last_style_score = style_score
        self.last_fashion_score = fashion_score

        # New weighted combination
        # Movement: 10%, Posture: 40%, Style (pose/face): 40%, Fashion: 10%
        rizz = (
            movement_score * 0.10 +
            posture_score * 0.40 +
            style_score   * 0.40 +
            fashion_score * 0.10
        )
        
        # Clamp to 0-100
        rizz = max(0, min(100, rizz))
        
        self.rizz_history.append(rizz)
        # Smooth the rizz score
        if len(self.rizz_history) > 1:
            self.rizz_score = np.mean(list(self.rizz_history))
        else:
            self.rizz_score = rizz
        
        return self.rizz_score
    
    def draw_rizz_meter(self, frame, head_x, head_y, rizz_score):
        """
        Draw the rizz meter above the person's head.
        
        Args:
            frame: Video frame
            head_x, head_y: Head top position coordinates (already at top of head)
            rizz_score: Current rizz score (0-100)
        """
        # Calculate meter position (above head - head_y is already at top of head)
        meter_width = 150
        meter_height = 30
        meter_x = int(head_x - meter_width // 2)
        # Position meter well above head (use frame height for proportional spacing)
        h, w = frame.shape[:2]
        spacing_above_head = int(h * 0.08)  # 8% of frame height above head
        meter_y = int(head_y - spacing_above_head)
        
        # Ensure meter is within frame bounds
        h, w = frame.shape[:2]
        meter_x = max(0, min(w - meter_width, meter_x))
        meter_y = max(meter_height, min(h, meter_y))
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (meter_x, meter_y - meter_height),
                     (meter_x + meter_width, meter_y),
                     (0, 0, 0), -1)
        cv2.rectangle(frame,
                     (meter_x, meter_y - meter_height),
                     (meter_x + meter_width, meter_y),
                     (255, 255, 255), 2)
        
        # Calculate color based on rizz score (green to yellow to red)
        if rizz_score < 33:
            color = (0, 0, 255)  # Red
        elif rizz_score < 66:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Draw filled bar
        fill_width = int((rizz_score / 100) * meter_width)
        cv2.rectangle(frame,
                     (meter_x + 2, meter_y - meter_height + 2),
                     (meter_x + fill_width - 2, meter_y - 2),
                     color, -1)
        
        # Draw text
        text = f"RIZZ: {int(rizz_score)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = meter_x + (meter_width - text_size[0]) // 2
        text_y = meter_y - meter_height // 2 + text_size[1] // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Draw sparkles or emoji for high rizz (visual flair)
        if rizz_score > 80:
            # Draw stars around meter
            star_points = [
                (meter_x - 10, meter_y - 10),
                (meter_x + meter_width + 10, meter_y - 10),
                (meter_x + meter_width // 2, meter_y - meter_height - 15)
            ]
            for point in star_points:
                cv2.circle(frame, point, 3, (255, 255, 0), -1)
    
    def get_head_position(self, landmarks):
        """
        Get the head position (top of head) from pose landmarks.
        Uses the distance from eyes to nose to estimate head height and top position.
        
        Args:
            landmarks: Pose landmarks (list from Tasks API)
            
        Returns:
            (x, y) tuple of head top position, or None if not detected
        """
        if not landmarks or len(landmarks) == 0:
            return None
        
        # Get head landmarks
        nose = self.get_landmark(landmarks, self.NOSE)
        left_eye = self.get_landmark(landmarks, self.LEFT_EYE)
        right_eye = self.get_landmark(landmarks, self.RIGHT_EYE)
        
        if not nose:
            return None
        
        # Calculate center x from eyes or use nose
        if left_eye and right_eye:
            avg_x = (left_eye.x + right_eye.x) / 2
            # Use the higher eye (smaller y value) as reference
            eye_y = min(left_eye.y, right_eye.y)
        else:
            avg_x = nose.x
            eye_y = nose.y
        
        # Estimate head height: distance from eye level to nose
        # Then add similar distance above eyes to get to top of head
        if nose and (left_eye or right_eye):
            eye_nose_distance = abs(nose.y - eye_y)
            # Top of head is approximately 1.5-2x the eye-nose distance above the eyes
            head_top_offset = eye_nose_distance * 1.8
            top_of_head_y = eye_y - head_top_offset
        else:
            # Fallback: use nose position and estimate upward
            top_of_head_y = nose.y - 0.08  # Rough estimate: 8% of frame height above nose
        
        return (avg_x, top_of_head_y)
    
    def draw_pose_landmarks(self, frame, landmarks):
        """
        Draw pose landmarks and connections on the frame.
        
        Args:
            frame: Video frame (BGR format)
            landmarks: Pose landmarks list from Tasks API
        """
        if not landmarks or len(landmarks) == 0:
            return
        
        h, w = frame.shape[:2]
        
        # Draw connections (skeleton lines)
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                # Check visibility if available
                if hasattr(start_landmark, 'visibility') and hasattr(end_landmark, 'visibility'):
                    if start_landmark.visibility < 0.5 or end_landmark.visibility < 0.5:
                        continue
                
                # Convert normalized coordinates to pixel coordinates
                start_x = int(start_landmark.x * w)
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w)
                end_y = int(end_landmark.y * h)
                
                # Draw connection line
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw all landmarks as circles
        for idx, landmark in enumerate(landmarks):
            if hasattr(landmark, 'visibility') and landmark.visibility < 0.5:
                continue
            
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Different colors for different parts
            if idx in [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR]:
                color = (255, 0, 0)  # Red for face
                radius = 4
            elif idx in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP]:
                color = (0, 255, 255)  # Yellow for torso
                radius = 5
            elif idx in [self.LEFT_WRIST, self.RIGHT_WRIST]:
                color = (255, 255, 0)  # Cyan for hands
                radius = 5
            elif idx in [self.LEFT_ANKLE, self.RIGHT_ANKLE]:
                color = (255, 0, 255)  # Magenta for feet
                radius = 5
            else:
                color = (0, 255, 0)  # Green for other points
                radius = 3
            
            cv2.circle(frame, (x, y), radius, color, -1)
    
    def run(self):
        """
        Main loop to run the rizz tracker.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        print("\n" + "="*60)
        print("Starting Rizz Tracker...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'v' - Toggle pose visualization (landmarks/skeleton)")
        print("  't' - Toggle tips box")
        print("="*60 + "\n")
        
        fps_counter = 0
        fps_start_time = time.time()
        fps_display = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Pose detection using Tasks API
            detection_result = self.pose_landmarker.detect(mp_image)
            
            # Extract landmarks (first person if multiple detected)
            landmarks = None
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]  # Get first person's landmarks
            
            # Fashion detection with YOLO (run less frequently for performance)
            detections = None
            if self.yolo_model and fps_counter % 5 == 0:  # Run every 5th frame
                detections = self.yolo_model(frame, verbose=False)
            
            # Calculate rizz score and draw visualizations
            if landmarks:
                rizz_score = self.calculate_rizz_score(landmarks, detections)
                
                # Draw pose landmarks/skeleton if enabled
                if self.show_pose_visualization:
                    self.draw_pose_landmarks(frame, landmarks)
                
                # Get head position (top of head)
                head_pos = self.get_head_position(landmarks)
                
                if head_pos:
                    # Convert normalized coordinates to pixel coordinates
                    h, w = frame.shape[:2]
                    head_x = int(head_pos[0] * w)
                    head_y = int(head_pos[1] * h)
                    
                    # Draw rizz meter above head (positioned from top of head)
                    self.draw_rizz_meter(frame, head_x, head_y, rizz_score)
            else:
                # No person detected, reset or show default
                rizz_score = 0.0
                cv2.putText(frame, "No person detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Generate tips based on latest sub-scores
            tips = self.get_rizz_tips(
                self.last_movement_score,
                self.last_posture_score,
                self.last_style_score,
                self.last_fashion_score,
            )

            # Draw tips box (left side)
            self.draw_rizz_tips_box(frame, tips)

            # Display FPS and controls
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                fps_display = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            # Display info overlay
            h, w = frame.shape[:2]
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show visualization/tips toggle status
            viz_status = "ON" if self.show_pose_visualization else "OFF"
            cv2.putText(frame, f"Pose Viz: {viz_status} (Press 'v')", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Rizz Tracker', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                self.show_pose_visualization = not self.show_pose_visualization
                print(f"Pose visualization: {'ON' if self.show_pose_visualization else 'OFF'}")
            elif key == ord('t'):
                self.show_tips_box = not self.show_tips_box
                print(f"Tips box: {'ON' if self.show_tips_box else 'OFF'}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose_landmarker.close()
        print("\nRizz Tracker stopped.")


if __name__ == "__main__":
    tracker = RizzTracker(camera_index=0)
    tracker.run()
