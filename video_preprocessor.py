import os
import cv2
import json
import glob
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from PIL import Image, ImageDraw

class SignLanguagePreprocessor:
    def __init__(self, data_dir="./data", batch_size=50):
        """
        Initialize the sign language video preprocessor.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.videos_dir = os.path.join(self.data_dir, "videos")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.features_dir = os.path.join(self.data_dir, "features")
        self.batch_size = batch_size

        # Create output directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

        # Initialize MediaPipe components
        self.mp_holistic = mp.solutions.holistic

        # Load sign language metadata from JSON files
        self.sign_data = self._load_sign_data()

    def _load_sign_data(self):
        """
        Load sign language metadata from JSON files.
        """
        sign_data = {}

        # Process ASL descriptions
        asl_file = os.path.join(self.data_dir, "asl_descriptions.json")
        if os.path.exists(asl_file):
            try:
                with open(asl_file, 'r') as f:
                    asl_data = json.load(f)
                    for sign, description in asl_data.items():
                        if sign not in sign_data:
                            sign_data[sign] = {"description": description, "videos": []}
            except Exception as e:
                print(f"Error processing ASL descriptions: {e}")

        # Process WLASL data to get video references
        wlasl_file = os.path.join(self.data_dir, "WLASL_v0.3.json")
        if os.path.exists(wlasl_file):
            try:
                with open(wlasl_file, 'r') as f:
                    wlasl_data = json.load(f)
                    for item in wlasl_data:
                        if "gloss" in item:
                            gloss = item["gloss"]
                            if gloss not in sign_data:
                                sign_data[gloss] = {"description": "", "videos": []}
                            
                            # Extract video references
                            for instance in item.get("instances", []):
                                if "video" in instance:
                                    video_id = instance["video"].split("/")[-1]
                                    sign_data[gloss]["videos"].append(video_id)
            except Exception as e:
                print(f"Error processing WLASL data: {e}")

        return sign_data

    def extract_landmarks(self, video_path, max_frames=30):
        """
        Extract body, hand, and face landmarks from a video using MediaPipe.
        """
        landmarks_data = []
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        ) as holistic:
            while cap.isOpened() and frame_count < max_frames:
                success, image = cap.read()
                if not success:
                    break
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process image with MediaPipe
                results = holistic.process(image_rgb)
                
                # Extract landmarks for this frame
                frame_landmarks = {
                    "frame": frame_count,
                    "pose": self._extract_pose_landmarks(results.pose_landmarks),
                    "left_hand": self._extract_hand_landmarks(results.left_hand_landmarks),
                    "right_hand": self._extract_hand_landmarks(results.right_hand_landmarks),
                    "face": self._extract_face_landmarks(results.face_landmarks)
                }
                
                landmarks_data.append(frame_landmarks)
                frame_count += 1
        
        cap.release()
        return landmarks_data

    def _extract_pose_landmarks(self, landmarks):
        """
        Extract pose landmarks from MediaPipe results.
        """
        if not landmarks:
            return None
        
        pose_points = []
        for i, point in enumerate(landmarks.landmark):
            pose_points.append({
                "idx": i,
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "visibility": point.visibility
            })
        
        return pose_points

    def _extract_hand_landmarks(self, landmarks):
        """
        Extract hand landmarks from MediaPipe results.
        """
        if not landmarks:
            return None
        
        hand_points = []
        for i, point in enumerate(landmarks.landmark):
            hand_points.append({
                "idx": i,
                "x": point.x,
                "y": point.y,
                "z": point.z
            })
        
        return hand_points

    def _extract_face_landmarks(self, landmarks):
        """
        Extract face landmarks from MediaPipe results.
        """
        if not landmarks:
            return None
        
        important_indices = [
            # Eyes
            33, 133, 160, 144, 158, 153,
            # Eyebrows
            65, 105,
            # Nose
            1, 2, 3, 4, 5,
            # Mouth
            10, 11, 12, 13, 14
        ]
        
        face_points = []
        
        for i in important_indices:
            point = landmarks.landmark[i]
            face_points.append({
                "idx": i,
                "x": point.x,
                "y": point.y,
                "z": point.z
            })
        
        return face_points

    def normalize_landmarks(self, landmarks_data):
        """
        Normalize landmarks to be invariant to position and scale.
        """
        normalized_data = []

        for frame_data in landmarks_data:
            pose = frame_data.get("pose")
            
            if not pose:
                normalized_data.append(frame_data)
                continue
            
            left_shoulder = next((p for p in pose if p["idx"] == 11), None)
            right_shoulder = next((p for p in pose if p["idx"] == 12), None)

            if not left_shoulder or not right_shoulder:
                normalized_data.append(frame_data)
                continue
            
            ref_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
            ref_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            
            scale = np.sqrt(
                (left_shoulder["x"] - right_shoulder["x"])**2 +
                (left_shoulder["y"] - right_shoulder["y"])**2
            )
            
            norm_frame = {"frame": frame_data["frame"]}
            
            for part in ["pose", "left_hand", "right_hand", "face"]:
                if frame_data.get(part):
                    norm_frame[part] = []
                    for point in frame_data[part]:
                        norm_point = {
                            **point,
                            "x": (point["x"] - ref_x) / scale,
                            "y": (point["y"] - ref_y) / scale,
                        }
                        norm_frame[part].append(norm_point)
            
            normalized_data.append(norm_frame)

        return normalized_data

    def process_video_batch(self):
      pass
