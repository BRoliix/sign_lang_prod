# src/preprocessing/video_processor.py
import cv2
import os
import numpy as np
from tqdm import tqdm
import glob

class VideoProcessor:
    def __init__(self, input_dir="data/raw", output_dir="data/processed/videos"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process_video(self, video_path, output_path=None, resize=(224, 224), fps=30):
        """Process a video file (resize, normalize, adjust fps)"""
        # Generate output path if not provided
        if output_path is None:
            video_name = os.path.basename(video_path)
            output_path = os.path.join(self.output_dir, video_name)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resize)
        
        # Process the video
        frames_to_keep = []
        if original_fps > 0:
            # Calculate which frames to keep to achieve target fps
            if original_fps > fps:
                # Skip frames
                frames_to_keep = [int(i * original_fps / fps) for i in range(int(frame_count * fps / original_fps))]
            else:
                # Keep all frames and potentially duplicate some
                frames_to_keep = list(range(frame_count))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if not frames_to_keep or frame_idx in frames_to_keep:
                # Resize frame
                resized_frame = cv2.resize(frame, resize)
                
                # Write the frame
                out.write(resized_frame)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
        
        return output_path
    
    def find_all_videos(self):
        """Find all video files in the input directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.input_dir, '**', f'*{ext}'), recursive=True))
        
        return video_files
    
    def process_all_videos(self):
        """Process all videos found in the input directory"""
        video_files = self.find_all_videos()
        print(f"Found {len(video_files)} video files")
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                video_name = os.path.basename(video_path)
                output_path = os.path.join(self.output_dir, video_name)
                self.process_video(video_path, output_path)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
