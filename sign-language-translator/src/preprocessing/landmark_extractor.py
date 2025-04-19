# src/preprocessing/landmark_extractor.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import glob

class LandmarkExtractor:
    def __init__(self, input_dir="data/processed/videos", output_dir="data/processed/landmarks"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hands model
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize holistic model for pose detection
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, video_path, output_path=None):
        """Extract hand and pose landmarks from a video"""
        # Generate output path if not provided
        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.output_dir, f"{video_name}_landmarks.npy")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        landmarks_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Holistic
            holistic_results = self.holistic.process(frame_rgb)
            
            # Extract landmarks
            frame_landmarks = []
            
            # Extract hand landmarks
            if holistic_results.left_hand_landmarks:
                for landmark in holistic_results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                # Add zeros for missing left hand landmarks
                frame_landmarks.extend([0] * (21 * 3))
            
            if holistic_results.right_hand_landmarks:
                for landmark in holistic_results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                # Add zeros for missing right hand landmarks
                frame_landmarks.extend([0] * (21 * 3))
            
            # Extract pose landmarks (focus on upper body)
            if holistic_results.pose_landmarks:
                # Upper body landmarks (11-23)
                for i in range(11, 23):
                    landmark = holistic_results.pose_landmarks.landmark[i]
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                # Add zeros for missing pose landmarks
                frame_landmarks.extend([0] * (12 * 3))
            
            landmarks_sequence.append(frame_landmarks)
        
        # Release resources
        cap.release()
        
        # Save landmarks
        landmarks_array = np.array(landmarks_sequence)
        np.save(output_path, landmarks_array)
        
        return output_path
    
    def find_all_videos(self):
        """Find all video files in the input directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.input_dir, f'*{ext}')))
        
        return video_files
    
    def extract_from_all_videos(self):
        """Extract landmarks from all videos in the input directory"""
        video_files = self.find_all_videos()
        print(f"Found {len(video_files)} video files")
        
        for video_path in tqdm(video_files, desc="Extracting landmarks"):
            try:
                self.extract_landmarks(video_path)
            except Exception as e:
                print(f"Error extracting landmarks from {video_path}: {str(e)}")
    
    def create_dataset_csv(self, annotations_dir="data/raw"):
        """Create a CSV file mapping landmark files to their annotations"""
        # Find all landmark files
        landmark_files = glob.glob(os.path.join(self.output_dir, "*_landmarks.npy"))
        
        # Try to find annotation files
        annotation_files = {}
        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if file.endswith(('.txt', '.csv')):
                    file_path = os.path.join(root, file)
                    # Use the filename without extension as the key
                    key = os.path.splitext(file)[0]
                    annotation_files[key] = file_path
        
        # Create dataset entries
        dataset = []
        for landmark_file in landmark_files:
            video_id = os.path.basename(landmark_file).split('_landmarks.npy')[0]
            
            # Try to find a matching annotation
            annotation = ""
            for key, annotation_path in annotation_files.items():
                if video_id in key or key in video_id:
                    try:
                        with open(annotation_path, 'r') as f:
                            annotation = f.read().strip()
                        break
                    except:
                        pass
            
            dataset.append({
                'video_id': video_id,
                'landmark_path': landmark_file,
                'annotation': annotation
            })
        
        # Save as CSV
        df = pd.DataFrame(dataset)
        csv_path = os.path.join(self.output_dir, "dataset.csv")
        df.to_csv(csv_path, index=False)
        print(f"Dataset CSV saved to {csv_path}")
        
        return csv_path
