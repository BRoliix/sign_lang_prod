# movement_detection.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import json

class MovementDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    
    def extract_keypoints(self, frame):
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image and detect landmarks
        results = self.holistic.process(image)
        
        # Extract landmarks
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                        results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in 
                        results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in 
                      results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in 
                      results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])
    
    def visualize_landmarks(self, frame, results):
        # Draw landmarks on the frame
        image = frame.copy()
        image.flags.writeable = True
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Draw face landmarks
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        
        # Draw hand landmarks
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
        
        return image
    
    def process_video(self, video_path, output_dir, visualize=False):
        cap = cv2.VideoCapture(video_path)
        frames_keypoints = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract keypoints
            keypoints = self.extract_keypoints(frame)
            frames_keypoints.append(keypoints)
            
            # Visualize if needed
            if visualize:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image)
                annotated_frame = self.visualize_landmarks(frame, results)
                cv2.imshow('MediaPipe Holistic', annotated_frame)
                
                if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save keypoints
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        video_name = os.path.basename(video_path).split('.')[0]
        np.save(os.path.join(output_dir, f"{video_name}_keypoints.npy"), np.array(frames_keypoints))
        
        return np.array(frames_keypoints)
