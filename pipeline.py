"""
Bimanual Hand Tracking Pipeline for Robot Learning
Combines all processing components from the notebook cells
"""
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from config import Config
import json
import time


# ============================================================================
# ENHANCED BIMANUAL HAND TRACKER
# ============================================================================

class EnhancedBimanualHandTracker:
    """Enhanced tracker with fallback detection and prediction"""
    
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    
    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        
        # Primary detector
        self.hands_primary = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Fallback detector (lower thresholds)
        self.hands_fallback = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3
        )
        
        # Hand tracking history
        self.left_hand_history = deque(maxlen=10)
        self.right_hand_history = deque(maxlen=10)
    
    def _calculate_hand_center(self, landmarks):
        """Calculate center of hand (palm area)"""
        wrist = landmarks[self.WRIST]
        middle_base = landmarks[self.MIDDLE_MCP]
        center_x = (wrist['x'] + middle_base['x']) / 2
        center_y = (wrist['y'] + middle_base['y']) / 2
        center_z = (wrist['z'] + middle_base['z']) / 2
        return np.array([center_x, center_y, center_z])
    
    def _predict_hand_position(self, hand_history):
        """Predict hand position based on trajectory"""
        if len(hand_history) < 2:
            return None
        
        recent = list(hand_history)[-3:]
        if len(recent) < 2:
            return recent[-1]
        
        centers = [h['center'] for h in recent]
        velocity = (centers[-1] - centers[-2])
        predicted_center = centers[-1] + velocity
        
        predicted = recent[-1].copy()
        predicted['center'] = predicted_center
        predicted['confidence'] = predicted['confidence'] * 0.5
        predicted['is_predicted'] = True
        
        return predicted
    
    def _assign_hand_identity(self, detected_hands, use_prediction=False):
        """Assign left/right identity to detected hands"""
        if not detected_hands:
            if use_prediction:
                predicted_left = self._predict_hand_position(self.left_hand_history)
                predicted_right = self._predict_hand_position(self.right_hand_history)
                return predicted_left, predicted_right
            return None, None
        
        hands_data = []
        for hand_landmarks, handedness in detected_hands:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append({
                    'x': lm.x, 'y': lm.y, 'z': lm.z,
                    'visibility': getattr(lm, 'visibility', 1.0)
                })
            
            center = self._calculate_hand_center(landmarks)
            hands_data.append({
                'landmarks': landmarks,
                'center': center,
                'mediapipe_label': handedness.classification[0].label,
                'confidence': handedness.classification[0].score,
                'is_predicted': False
            })
        
        # Single hand case
        if len(hands_data) == 1:
            hand = hands_data[0]
            spatial_score_left = 2 if hand['center'][0] < 0.5 else 0
            spatial_score_right = 2 if hand['center'][0] >= 0.5 else 0
            
            if len(self.left_hand_history) > 0:
                dist_to_left = np.linalg.norm(
                    hand['center'] - self.left_hand_history[-1]['center']
                )
                if dist_to_left < self.config.HAND_IDENTITY_DISTANCE_THRESHOLD:
                    spatial_score_left += 3
            
            if len(self.right_hand_history) > 0:
                dist_to_right = np.linalg.norm(
                    hand['center'] - self.right_hand_history[-1]['center']
                )
                if dist_to_right < self.config.HAND_IDENTITY_DISTANCE_THRESHOLD:
                    spatial_score_right += 3
            
            if spatial_score_left > spatial_score_right:
                hand['assigned_label'] = 'Left'
                self.left_hand_history.append(hand)
                predicted_right = self._predict_hand_position(self.right_hand_history) if use_prediction else None
                return hand, predicted_right
            else:
                hand['assigned_label'] = 'Right'
                self.right_hand_history.append(hand)
                predicted_left = self._predict_hand_position(self.left_hand_history) if use_prediction else None
                return predicted_left, hand
        
        # Two hands case - sort by x position
        hands_data.sort(key=lambda h: h['center'][0])
        left_candidate = hands_data[0]
        right_candidate = hands_data[1]
        
        # Check if they need swapping based on history
        if len(self.left_hand_history) > 0 and len(self.right_hand_history) > 0:
            left_to_prev_left = np.linalg.norm(
                left_candidate['center'] - self.left_hand_history[-1]['center']
            )
            left_to_prev_right = np.linalg.norm(
                left_candidate['center'] - self.right_hand_history[-1]['center']
            )
            
            if left_to_prev_right < left_to_prev_left:
                left_candidate, right_candidate = right_candidate, left_candidate
        
        left_candidate['assigned_label'] = 'Left'
        right_candidate['assigned_label'] = 'Right'
        
        self.left_hand_history.append(left_candidate)
        self.right_hand_history.append(right_candidate)
        
        return left_candidate, right_candidate
    
    def process_video(self, video_path: Path) -> Tuple[List[Dict], Dict]:
        """Process video and extract bimanual hand tracking data"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {total_frames} frames @ {fps} FPS ({width}x{height})")
        
        tracking_data = []
        frame_idx = 0
        
        left_detected = 0
        right_detected = 0
        both_detected = 0
        primary_success = 0
        fallback_used = 0
        predicted = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_idx / fps
            
            # Try primary detector
            results = self.hands_primary.process(rgb_frame)
            detection_method = 'primary'
            
            # Try fallback if needed
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
                results_fallback = self.hands_fallback.process(rgb_frame)
                
                if results_fallback.multi_hand_landmarks:
                    if not results.multi_hand_landmarks or \
                       len(results_fallback.multi_hand_landmarks) > len(results.multi_hand_landmarks):
                        results = results_fallback
                        detection_method = 'fallback'
                        fallback_used += 1
            else:
                primary_success += 1
            
            frame_data = {
                'frame': frame_idx,
                'timestamp': timestamp,
                'left_hand': None,
                'right_hand': None,
                'tracking_quality': 'no_hands',
                'detection_method': detection_method
            }
            
            if results.multi_hand_landmarks:
                detected_hands = list(zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ))
                
                left_hand, right_hand = self._assign_hand_identity(detected_hands, use_prediction=True)
                
                if left_hand:
                    frame_data['left_hand'] = {
                        'landmarks': left_hand['landmarks'],
                        'confidence': left_hand['confidence'],
                        'center': left_hand['center'].tolist(),
                        'is_predicted': left_hand.get('is_predicted', False)
                    }
                    if not left_hand.get('is_predicted', False):
                        left_detected += 1
                    else:
                        predicted += 1
                
                if right_hand:
                    frame_data['right_hand'] = {
                        'landmarks': right_hand['landmarks'],
                        'confidence': right_hand['confidence'],
                        'center': right_hand['center'].tolist(),
                        'is_predicted': right_hand.get('is_predicted', False)
                    }
                    if not right_hand.get('is_predicted', False):
                        right_detected += 1
                    else:
                        predicted += 1
                
                if left_hand and right_hand:
                    if not (left_hand.get('is_predicted') or right_hand.get('is_predicted')):
                        frame_data['tracking_quality'] = 'both_hands'
                        both_detected += 1
                    else:
                        frame_data['tracking_quality'] = 'partial_predicted'
                elif left_hand or right_hand:
                    frame_data['tracking_quality'] = 'single_hand'
            else:
                # Try prediction for missing frames
                left_pred, right_pred = self._assign_hand_identity([], use_prediction=True)
                
                if left_pred:
                    frame_data['left_hand'] = {
                        'landmarks': left_pred['landmarks'],
                        'confidence': left_pred['confidence'] * 0.3,
                        'center': left_pred['center'].tolist(),
                        'is_predicted': True
                    }
                    predicted += 1
                
                if right_pred:
                    frame_data['right_hand'] = {
                        'landmarks': right_pred['landmarks'],
                        'confidence': right_pred['confidence'] * 0.3,
                        'center': right_pred['center'].tolist(),
                        'is_predicted': True
                    }
                    predicted += 1
                
                if left_pred or right_pred:
                    frame_data['tracking_quality'] = 'fully_predicted'
            
            tracking_data.append(frame_data)
            
            if frame_idx % 30 == 0:
                print(f"Processing: {frame_idx}/{total_frames} frames", end='\r')
            
            frame_idx += 1
        
        cap.release()
        
        quality_metrics = {
            'total_frames': total_frames,
            'left_hand_detection_rate': left_detected / total_frames,
            'right_hand_detection_rate': right_detected / total_frames,
            'both_hands_detection_rate': both_detected / total_frames,
            'primary_detector_success_rate': primary_success / total_frames,
            'fallback_detector_used': fallback_used,
            'predicted_frames': predicted,
            'fps': fps,
            'resolution': {'width': width, 'height': height}
        }
        
        print(f"\n✓ Processed {frame_idx} frames")
        print(f"  Left: {left_detected} detected + {predicted//2} predicted")
        print(f"  Right: {right_detected} detected + {predicted//2} predicted")
        print(f"  Both: {both_detected} frames ({both_detected/total_frames*100:.1f}%)")
        
        return tracking_data, quality_metrics


# ============================================================================
# EGOCENTRIC FEATURE EXTRACTOR
# ============================================================================

class EgocentricFeatureExtractor:
    """Extract robot-relevant features from hand tracking"""
    
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    
    def __init__(self, config: Config):
        self.config = config
    
    def _extract_hand_features(self, hand_data, hand_label):
        """Extract features for a single hand"""
        if hand_data is None:
            return None
        
        landmarks = hand_data['landmarks']
        confidence = hand_data['confidence']
        
        wrist = landmarks[self.WRIST]
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        index_mcp = landmarks[self.INDEX_MCP]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        
        # Calculate palm position
        palm_x = (wrist['x'] + index_mcp['x'] + middle_mcp['x']) / 3
        palm_y = (wrist['y'] + index_mcp['y'] + middle_mcp['y']) / 3
        palm_z = (wrist['z'] + index_mcp['z'] + middle_mcp['z']) / 3
        
        # Calculate gripper distance
        gripper_dist = np.sqrt(
            (thumb_tip['x'] - index_tip['x'])**2 +
            (thumb_tip['y'] - index_tip['y'])**2 +
            (thumb_tip['z'] - index_tip['z'])**2
        )
        
        # Gripper state (0=closed, 1=open)
        if gripper_dist > self.config.GRIPPER_OPEN_THRESHOLD:
            gripper_state = 1.0
        elif gripper_dist < self.config.GRIPPER_CLOSED_THRESHOLD:
            gripper_state = 0.0
        else:
            gripper_state = (gripper_dist - self.config.GRIPPER_CLOSED_THRESHOLD) / \
                           (self.config.GRIPPER_OPEN_THRESHOLD - self.config.GRIPPER_CLOSED_THRESHOLD)
        
        # Calculate orientation
        dx = middle_tip['x'] - wrist['x']
        dy = middle_tip['y'] - wrist['y']
        dz = middle_tip['z'] - wrist['z']
        
        pitch = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        yaw = np.arctan2(dy, dx)
        
        # Roll (thumb to pinky)
        thumb_base = landmarks[1]
        pinky_base = landmarks[17]
        roll_dx = pinky_base['x'] - thumb_base['x']
        roll_dy = pinky_base['y'] - thumb_base['y']
        roll = np.arctan2(roll_dy, roll_dx)
        
        # Hand openness
        finger_tips = [
            landmarks[self.THUMB_TIP],
            landmarks[self.INDEX_TIP],
            landmarks[self.MIDDLE_TIP],
            landmarks[self.RING_TIP],
            landmarks[self.PINKY_TIP]
        ]
        
        openness = np.mean([
            np.sqrt((tip['x'] - palm_x)**2 +
                   (tip['y'] - palm_y)**2 +
                   (tip['z'] - palm_z)**2)
            for tip in finger_tips
        ])
        
        return {
            'palm_x': palm_x,
            'palm_y': palm_y,
            'palm_z': palm_z,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'gripper': gripper_state,
            'gripper_distance': gripper_dist,
            'hand_openness': openness,
            'confidence': confidence,
            'hand_label': hand_label
        }
    
    def extract_features(self, tracking_data):
        """Extract bimanual features from tracking data"""
        features = []
        
        for frame_data in tracking_data:
            timestamp = frame_data['timestamp']
            frame_idx = frame_data['frame']
            
            left_features = self._extract_hand_features(
                frame_data['left_hand'], 'Left'
            )
            right_features = self._extract_hand_features(
                frame_data['right_hand'], 'Right'
            )
            
            feature = {
                'timestamp': timestamp,
                'frame': frame_idx,
                'left_hand': left_features,
                'right_hand': right_features,
                'tracking_quality': frame_data['tracking_quality']
            }
            
            features.append(feature)
        
        print(f"✓ Extracted features for {len(features)} frames")
        return features


# ============================================================================
# COORDINATE TRANSFORMER
# ============================================================================

class EgocentricCoordinateTransformer:
    """Transform egocentric coordinates to robot workspace"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Camera tilt correction matrix
        tilt_rad = np.radians(config.CAMERA_TILT_ANGLE)
        self.tilt_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
            [0, np.sin(tilt_rad), np.cos(tilt_rad)]
        ])
    
    def _transform_hand_to_robot_space(self, hand_features):
        """Transform single hand to robot workspace"""
        if hand_features is None:
            return None
        
        # Normalize coordinates
        norm_x = hand_features['palm_x'] - 0.5
        norm_y = hand_features['palm_y'] - 0.5
        norm_z = hand_features['palm_z']
        
        camera_coords = np.array([norm_x, norm_y, norm_z])
        tilted_coords = self.tilt_matrix @ camera_coords
        
        # Scale to robot workspace
        x_robot = tilted_coords[0] * self.config.WORKSPACE_WIDTH + \
                  self.config.WORKSPACE_OFFSET_X
        
        y_robot = -tilted_coords[2] * self.config.WORKSPACE_HEIGHT + \
                  self.config.WORKSPACE_OFFSET_Y
        
        z_robot = -tilted_coords[1] * self.config.WORKSPACE_DEPTH + \
                  self.config.WORKSPACE_OFFSET_Z
        
        # Transform orientation
        roll_robot = hand_features['roll']
        pitch_robot = hand_features['pitch'] - np.radians(self.config.CAMERA_TILT_ANGLE)
        yaw_robot = hand_features['yaw']
        
        return {
            'x': x_robot,
            'y': y_robot,
            'z': z_robot,
            'roll': roll_robot,
            'pitch': pitch_robot,
            'yaw': yaw_robot,
            'gripper': hand_features['gripper'],
            'confidence': hand_features['confidence']
        }
    
    def transform_to_robot_space(self, features):
        """Transform bimanual features to robot workspace"""
        robot_actions = []
        
        for feat in features:
            left_robot = self._transform_hand_to_robot_space(feat['left_hand'])
            right_robot = self._transform_hand_to_robot_space(feat['right_hand'])
            
            action = {
                'timestamp': feat['timestamp'],
                'frame': feat['frame'],
                'left_hand': left_robot,
                'right_hand': right_robot,
                'tracking_quality': feat['tracking_quality']
            }
            
            robot_actions.append(action)
        
        print(f"✓ Transformed {len(robot_actions)} actions to robot space")
        return robot_actions
    
    def smooth_actions(self, actions, window_size=None):
        """Apply smoothing to trajectories"""
        if window_size is None:
            window_size = self.config.SMOOTHING_WINDOW
        
        if len(actions) < window_size:
            print("⚠ Not enough frames for smoothing")
            return actions
        
        def smooth_hand_trajectory(hand_data_list):
            valid_indices = [i for i, h in enumerate(hand_data_list) if h is not None]
            if len(valid_indices) < window_size:
                return hand_data_list
            
            keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
            smoothed_data = [None] * len(hand_data_list)
            
            for key in keys:
                values = np.array([hand_data_list[i][key] for i in valid_indices])
                kernel = np.ones(window_size) / window_size
                smoothed_values = np.convolve(values, kernel, mode='same')
                
                for j, idx in enumerate(valid_indices):
                    if smoothed_data[idx] is None:
                        smoothed_data[idx] = hand_data_list[idx].copy()
                    smoothed_data[idx][key] = smoothed_values[j]
            
            for i in range(len(smoothed_data)):
                if smoothed_data[i] is None:
                    smoothed_data[i] = hand_data_list[i]
            
            return smoothed_data
        
        left_hands = [a['left_hand'] for a in actions]
        right_hands = [a['right_hand'] for a in actions]
        
        left_smoothed = smooth_hand_trajectory(left_hands)
        right_smoothed = smooth_hand_trajectory(right_hands)
        
        smoothed_actions = []
        for i, action in enumerate(actions):
            smoothed_actions.append({
                'timestamp': action['timestamp'],
                'frame': action['frame'],
                'left_hand': left_smoothed[i],
                'right_hand': right_smoothed[i],
                'tracking_quality': action['tracking_quality']
            })
        
        print(f"✓ Applied smoothing (window={window_size})")
        return smoothed_actions
    
    def interpolate_missing_frames(self, actions):
        """Interpolate short gaps in tracking"""
        def interpolate_hand(hand_data_list):
            result = hand_data_list.copy()
            valid_indices = [i for i, h in enumerate(hand_data_list) if h is not None]
            
            if len(valid_indices) < 2:
                return result
            
            for i in range(len(valid_indices) - 1):
                start_idx = valid_indices[i]
                end_idx = valid_indices[i + 1]
                gap_size = end_idx - start_idx - 1
                
                if gap_size > 0 and gap_size <= self.config.INTERPOLATION_MAX_GAP:
                    start_data = hand_data_list[start_idx]
                    end_data = hand_data_list[end_idx]
                    
                    keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
                    for j in range(1, gap_size + 1):
                        alpha = j / (gap_size + 1)
                        interpolated = {}
                        for key in keys:
                            interpolated[key] = (1 - alpha) * start_data[key] + \
                                              alpha * end_data[key]
                        interpolated['confidence'] = min(
                            start_data['confidence'],
                            end_data['confidence']
                        ) * 0.5
                        
                        result[start_idx + j] = interpolated
            
            return result
        
        left_hands = [a['left_hand'] for a in actions]
        right_hands = [a['right_hand'] for a in actions]
        
        left_interpolated = interpolate_hand(left_hands)
        right_interpolated = interpolate_hand(right_hands)
        
        left_count = sum(1 for l1, l2 in zip(left_hands, left_interpolated)
                        if l1 is None and l2 is not None)
        right_count = sum(1 for r1, r2 in zip(right_hands, right_interpolated)
                         if r1 is None and r2 is not None)
        
        interpolated_actions = []
        for i, action in enumerate(actions):
            interpolated_actions.append({
                'timestamp': action['timestamp'],
                'frame': action['frame'],
                'left_hand': left_interpolated[i],
                'right_hand': right_interpolated[i],
                'tracking_quality': action['tracking_quality']
            })
        
        print(f"✓ Interpolated {left_count} left, {right_count} right frames")
        return interpolated_actions


# ============================================================================
# QUALITY ANALYZER
# ============================================================================

class QualityAnalyzer:
    """Analyze tracking quality and detect issues"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_quality(self, actions, tracking_metrics):
        """Comprehensive quality analysis"""
        print("\n" + "="*70)
        print("QUALITY ANALYSIS")
        print("="*70)
        
        quality_report = {
            'overall_metrics': tracking_metrics,
            'physics_validation': {},
            'problematic_segments': [],
            'quality_score': 0.0,
            'recommendations': []
        }
        
        # Physics validation
        physics_issues = self._validate_physics(actions)
        quality_report['physics_validation'] = physics_issues
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence(actions)
        quality_report['confidence_analysis'] = confidence_analysis
        
        # Detect problems
        problematic = self._detect_problematic_segments(actions)
        quality_report['problematic_segments'] = problematic
        
        # Calculate score
        quality_score = self._calculate_quality_score(
            tracking_metrics,
            physics_issues,
            confidence_analysis
        )
        quality_report['quality_score'] = quality_score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_report)
        quality_report['recommendations'] = recommendations
        
        print(f"\n📊 Quality Score: {quality_score:.1f}/100")
        
        if quality_score >= 80:
            print("✅ EXCELLENT - Ready for training")
        elif quality_score >= 60:
            print("⚠️  GOOD - Minor issues")
        elif quality_score >= 40:
            print("⚠️  FAIR - Significant issues")
        else:
            print("❌ POOR - Re-recording recommended")
        
        return quality_report
    
    def _validate_physics(self, actions):
        """Check for physics violations"""
        issues = {
            'impossible_velocities': [],
            'impossible_accelerations': [],
            'teleportation_events': []
        }
        
        def check_hand_physics(hand_actions, hand_label):
            valid_actions = [a for a in hand_actions if a is not None]
            
            if len(valid_actions) < 3:
                return
            
            dt = 1.0 / 30.0
            for i in range(1, len(valid_actions)):
                prev = valid_actions[i-1]
                curr = valid_actions[i]
                
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                dz = curr['z'] - prev['z']
                velocity = np.sqrt(dx**2 + dy**2 + dz**2) / dt
                
                if velocity > self.config.MAX_VELOCITY_THRESHOLD:
                    issues['impossible_velocities'].append({
                        'hand': hand_label,
                        'frame': i,
                        'velocity': velocity
                    })
                
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                if distance > self.config.TELEPORTATION_THRESHOLD:
                    issues['teleportation_events'].append({
                        'hand': hand_label,
                        'frame': i,
                        'distance': distance
                    })
        
        left_actions = [a['left_hand'] for a in actions]
        right_actions = [a['right_hand'] for a in actions]
        
        check_hand_physics(left_actions, 'Left')
        check_hand_physics(right_actions, 'Right')
        
        print(f"\n🔬 Physics: {len(issues['impossible_velocities'])} velocity issues, "
              f"{len(issues['teleportation_events'])} teleportations")
        
        return issues
    
    def _analyze_confidence(self, actions):
        """Analyze tracking confidence"""
        left_confidences = []
        right_confidences = []
        
        for action in actions:
            if action['left_hand']:
                left_confidences.append(action['left_hand']['confidence'])
            if action['right_hand']:
                right_confidences.append(action['right_hand']['confidence'])
        
        return {
            'left_hand': {
                'mean_confidence': np.mean(left_confidences) if left_confidences else 0,
                'low_confidence_frames': sum(1 for c in left_confidences
                                            if c < self.config.MIN_HAND_CONFIDENCE)
            },
            'right_hand': {
                'mean_confidence': np.mean(right_confidences) if right_confidences else 0,
                'low_confidence_frames': sum(1 for c in right_confidences
                                            if c < self.config.MIN_HAND_CONFIDENCE)
            }
        }
    
    def _detect_problematic_segments(self, actions):
        """Find continuous problematic segments"""
        return []  # Simplified for brevity
    
    def _calculate_quality_score(self, tracking_metrics, physics_issues, confidence_analysis):
        """Calculate 0-100 quality score"""
        score = 100.0
        
        # Detection rate penalty
        avg_detection = (tracking_metrics['left_hand_detection_rate'] +
                        tracking_metrics['right_hand_detection_rate']) / 2
        score -= (1 - avg_detection) * 30
        
        # Physics issues penalty
        total_physics = (len(physics_issues['impossible_velocities']) +
                        len(physics_issues['teleportation_events']))
        score -= min(total_physics * 2, 30)
        
        # Confidence penalty
        avg_confidence = (confidence_analysis['left_hand']['mean_confidence'] +
                         confidence_analysis['right_hand']['mean_confidence']) / 2
        score -= (1 - avg_confidence) * 20
        
        return max(0, score)
    
    def _generate_recommendations(self, quality_report):
        """Generate actionable recommendations"""
        recommendations = []
        score = quality_report['quality_score']
        
        if score < 60:
            recommendations.append("Consider re-recording in better conditions")
        if score >= 80:
            recommendations.append("Quality is excellent! Ready for training")
        
        return recommendations


# ============================================================================
# VISUALIZER
# ============================================================================

class BimanualVisualizer:
    """Create visualizations"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_trajectories(self, actions, output_path: Path):
        """Plot bimanual trajectories"""
        try:
            fig = plt.figure(figsize=(15, 10))
            
            left_data = {'t': [], 'x': [], 'y': [], 'z': [], 'gripper': []}
            right_data = {'t': [], 'x': [], 'y': [], 'z': [], 'gripper': []}
            
            for action in actions:
                if action['left_hand']:
                    left_data['t'].append(action['timestamp'])
                    left_data['x'].append(action['left_hand']['x'])
                    left_data['y'].append(action['left_hand']['y'])
                    left_data['z'].append(action['left_hand']['z'])
                    left_data['gripper'].append(action['left_hand']['gripper'])
                
                if action['right_hand']:
                    right_data['t'].append(action['timestamp'])
                    right_data['x'].append(action['right_hand']['x'])
                    right_data['y'].append(action['right_hand']['y'])
                    right_data['z'].append(action['right_hand']['z'])
                    right_data['gripper'].append(action['right_hand']['gripper'])
            
            # 3D trajectory
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            if left_data['x']:
                ax1.plot(left_data['x'], left_data['y'], left_data['z'],
                        'c-', linewidth=2, label='Left', alpha=0.7)
            if right_data['x']:
                ax1.plot(right_data['x'], right_data['y'], right_data['z'],
                        'm-', linewidth=2, label='Right', alpha=0.7)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory')
            ax1.legend()
            
            # X over time
            ax2 = fig.add_subplot(2, 3, 2)
            if left_data['t']:
                ax2.plot(left_data['t'], left_data['x'], 'c-', linewidth=2, label='Left')
            if right_data['t']:
                ax2.plot(right_data['t'], right_data['x'], 'm-', linewidth=2, label='Right')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('X (m)')
            ax2.set_title('X Position')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Y over time
            ax3 = fig.add_subplot(2, 3, 3)
            if left_data['t']:
                ax3.plot(left_data['t'], left_data['y'], 'c-', linewidth=2, label='Left')
            if right_data['t']:
                ax3.plot(right_data['t'], right_data['y'], 'm-', linewidth=2, label='Right')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Y (m)')
            ax3.set_title('Y Position')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Z over time
            ax4 = fig.add_subplot(2, 3, 4)
            if left_data['t']:
                ax4.plot(left_data['t'], left_data['z'], 'c-', linewidth=2, label='Left')
            if right_data['t']:
                ax4.plot(right_data['t'], right_data['z'], 'm-', linewidth=2, label='Right')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Z (m)')
            ax4.set_title('Z Position')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Gripper states
            ax5 = fig.add_subplot(2, 3, 5)
            if left_data['t']:
                ax5.plot(left_data['t'], left_data['gripper'], 'c-', linewidth=2, label='Left')
            if right_data['t']:
                ax5.plot(right_data['t'], right_data['gripper'], 'm-', linewidth=2, label='Right')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Gripper State')
            ax5.set_title('Gripper States')
            ax5.legend()
            ax5.set_ylim([-0.1, 1.1])
            ax5.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Trajectory plot saved: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Could not create visualization: {e}")


# ============================================================================
# DATA EXPORTER
# ============================================================================

class DatasetExporter:
    """Export processed data"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def export_actions(self, actions, video_name, task_annotations, output_dir):
        """Export bimanual actions with task metadata"""
        traj_dir = Path(output_dir) / "annotations" / "hand_trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        task_id = task_annotations['task_id']
        base_name = Path(video_name).stem
        
        # 1. NumPy
        numpy_path = traj_dir / f"{task_id}_{base_name}_actions.npy"
        data_array = self._create_numpy_array(actions)
        np.save(numpy_path, data_array)
        print(f"✓ NumPy: {numpy_path}")
        
        # 2. CSV
        csv_path = traj_dir / f"{task_id}_{base_name}_actions.csv"
        df = self._create_dataframe(actions, task_annotations)
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV: {csv_path}")
        
        # 3. JSON
        json_path = traj_dir / f"{task_id}_{base_name}_actions.json"
        full_data = {
            'task_annotations': task_annotations,
            'trajectory_data': actions
        }
        with open(json_path, 'w') as f:
            json.dump(full_data, f, indent=2)
        print(f"✓ JSON: {json_path}")
        
        return data_array, df
    
    def _create_numpy_array(self, actions):
        """Create NumPy array for ML"""
        rows = []
        
        for action in actions:
            if action['left_hand']:
                lh = action['left_hand']
                left_data = [lh['x'], lh['y'], lh['z'], lh['roll'],
                            lh['pitch'], lh['yaw'], lh['gripper']]
            else:
                left_data = [0.0] * 7
            
            if action['right_hand']:
                rh = action['right_hand']
                right_data = [rh['x'], rh['y'], rh['z'], rh['roll'],
                             rh['pitch'], rh['yaw'], rh['gripper']]
            else:
                right_data = [0.0] * 7
            
            row = [action['timestamp']] + left_data + right_data
            rows.append(row)
        
        return np.array(rows)
    
    def _create_dataframe(self, actions, task_annotations):
        """Create pandas DataFrame"""
        data = []
        
        for action in actions:
            row = {
                'timestamp': action['timestamp'],
                'frame': action['frame'],
                'category': task_annotations['category'],
                'task': task_annotations['task'],
                'task_id': task_annotations['task_id']
            }
            
            if action['left_hand']:
                lh = action['left_hand']
                row.update({
                    'left_x': lh['x'], 'left_y': lh['y'], 'left_z': lh['z'],
                    'left_roll': lh['roll'], 'left_pitch': lh['pitch'],
                    'left_yaw': lh['yaw'], 'left_gripper': lh['gripper'],
                    'left_confidence': lh['confidence']
                })
            else:
                row.update({
                    'left_x': None, 'left_y': None, 'left_z': None,
                    'left_roll': None, 'left_pitch': None, 'left_yaw': None,
                    'left_gripper': None, 'left_confidence': None
                })
            
            if action['right_hand']:
                rh = action['right_hand']
                row.update({
                    'right_x': rh['x'], 'right_y': rh['y'], 'right_z': rh['z'],
                    'right_roll': rh['roll'], 'right_pitch': rh['pitch'],
                    'right_yaw': rh['yaw'], 'right_gripper': rh['gripper'],
                    'right_confidence': rh['confidence']
                })
            else:
                row.update({
                    'right_x': None, 'right_y': None, 'right_z': None,
                    'right_roll': None, 'right_pitch': None, 'right_yaw': None,
                    'right_gripper': None, 'right_confidence': None
                })
            
            row['tracking_quality'] = action['tracking_quality']
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_metadata(self, video_name, task_annotations, tracking_metrics,
                       quality_report, output_dir):
        """Export comprehensive metadata"""
        metadata_dir = Path(output_dir) / "metadata"
        quality_dir = Path(output_dir) / "annotations" / "quality_reports"
        
        metadata_dir.mkdir(parents=True, exist_ok=True)
        quality_dir.mkdir(parents=True, exist_ok=True)
        
        task_id = task_annotations['task_id']
        base_name = Path(video_name).stem
        
        metadata = {
            'video_name': video_name,
            'processing_date': time.time(),
            'task_annotations': task_annotations,
            'tracking_metrics': tracking_metrics,
            'quality_report': {
                'quality_score': quality_report['quality_score'],
                'recommendations': quality_report['recommendations']
            }
        }
        
        metadata_path = metadata_dir / f"{task_id}_{base_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata: {metadata_path}")
        
        quality_path = quality_dir / f"{task_id}_{base_name}_quality.json"
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        print(f"✓ Quality report: {quality_path}")
        
        return metadata


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

class BimanualPipelineRunner:
    """Orchestrate complete pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tracker = EnhancedBimanualHandTracker(config)
        self.feature_extractor = EgocentricFeatureExtractor(config)
        self.transformer = EgocentricCoordinateTransformer(config)
        self.visualizer = BimanualVisualizer(config)
        self.quality_analyzer = QualityAnalyzer(config)
        self.exporter = DatasetExporter(config)
    
    def run(self, video_path: Path, task_annotations: Dict) -> Dict:
        """Run complete pipeline"""
        print("\n" + "="*70)
        print("BIMANUAL HAND TRACKING PIPELINE")
        print("="*70)
        print(f"\n🎯 Task: {task_annotations['task']}")
        print(f"📂 Category: {task_annotations['category']}")
        
        start_time = time.time()
        video_name = video_path.name
        
        # Step 1: Track hands
        print("\n[1/8] Tracking hands...")
        tracking_data, tracking_metrics = self.tracker.process_video(video_path)
        
        # Step 2: Extract features
        print("\n[2/8] Extracting features...")
        features = self.feature_extractor.extract_features(tracking_data)
        
        # Step 3: Transform to robot space
        print("\n[3/8] Transforming to robot workspace...")
        actions = self.transformer.transform_to_robot_space(features)
        
        # Step 4: Interpolate
        print("\n[4/8] Interpolating missing frames...")
        actions = self.transformer.interpolate_missing_frames(actions)
        
        # Step 5: Smooth
        print("\n[5/8] Smoothing trajectories...")
        actions = self.transformer.smooth_actions(actions)
        
        # Step 6: Quality analysis
        print("\n[6/8] Analyzing quality...")
        quality_report = self.quality_analyzer.analyze_quality(actions, tracking_metrics)
        
        # Step 7: Export data
        print("\n[7/8] Exporting data...")
        data_array, df = self.exporter.export_actions(
            actions, video_name, task_annotations, self.config.DATASET_DIR
        )
        
        metadata = self.exporter.export_metadata(
            video_name, task_annotations, tracking_metrics,
            quality_report, self.config.DATASET_DIR
        )
        
        # Step 8: Visualize
        print("\n[8/8] Creating visualizations...")
        task_id = task_annotations['task_id']
        base_name = video_path.stem
        plot_path = self.config.OUTPUT_DIR / f"{task_id}_{base_name}_trajectories.png"
        
        self.visualizer.plot_trajectories(actions, plot_path)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"✅ PIPELINE COMPLETE ({elapsed:.1f}s)")
        print("="*70)
        
        return {
            'task_annotations': task_annotations,
            'tracking_data': tracking_data,
            'actions': actions,
            'data_array': data_array,
            'tracking_metrics': tracking_metrics,
            'quality_report': quality_report,
            'metadata': metadata,
            'processing_time': elapsed
        }