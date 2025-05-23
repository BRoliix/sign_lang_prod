# process_openasl.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import json
import concurrent.futures
import argparse

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    
    def process_video(self, video_path, output_dir, visualize=False):
        cap = cv2.VideoCapture(video_path)
        frames_keypoints = []
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
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
        if visualize:
            cv2.destroyAllWindows()
        
        # Save keypoints
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        video_name = os.path.basename(video_path).split('.')[0]
        # Remove the underscore prefix if it exists
        if video_name.startswith('_-'):
            video_name = video_name[2:]
            
        np.save(os.path.join(output_dir, f"{video_name}_keypoints.npy"), np.array(frames_keypoints))
        
        return np.array(frames_keypoints)

# Move this function outside the class and don't pass the detector
def process_video_file(args):
    video_path, output_dir = args
    try:
        print(f"Starting to process: {os.path.basename(video_path)}")
        # Create a new detector instance inside the function
        detector = MovementDetector()
        detector.process_video(video_path, output_dir)
        print(f"Finished processing: {os.path.basename(video_path)}")
        return True
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False

def find_video_file(videos_dir, video_id):
    """Find the video file with or without underscore prefix."""
    # Try direct match first
    direct_path = os.path.join(videos_dir, f"{video_id}.mp4")
    if os.path.exists(direct_path):
        return direct_path
    
    # Try with underscore prefix
    underscore_path = os.path.join(videos_dir, f"_-{video_id}.mp4")
    if os.path.exists(underscore_path):
        return underscore_path
    
    # Try finding a file that contains the video_id (without timestamp)
    base_id = video_id.split('-')[0]  # Get the base ID without timestamps
    for filename in os.listdir(videos_dir):
        if filename.endswith('.mp4') and base_id in filename:
            return os.path.join(videos_dir, filename)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Process OpenASL videos')
    parser.add_argument('--tsv_path', type=str, default='data/raw/openasl/tsv/openasl-v1.0.tsv', 
                        help='Path to OpenASL TSV file')
    parser.add_argument('--bbox_path', type=str, default='data/raw/openasl/bbox/bbox-v1.0.json', 
                        help='Path to bbox JSON file')
    parser.add_argument('--videos_dir', type=str, default='data/raw/openasl/videos', 
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='data/processed/keypoints', 
                        help='Output directory for keypoints')
    parser.add_argument('--num_workers', type=int, default=1,  # Reduced to 1 for Mac M3
                        help='Number of parallel workers')
    parser.add_argument('--list_only', action='store_true',
                        help='Only list available videos without processing')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (for testing)')
    
    args = parser.parse_args()
    
    # Load TSV file
    df = pd.read_csv(args.tsv_path, sep='\t', low_memory=False)
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Load bbox file
    with open(args.bbox_path, 'r') as f:
        bbox_data = json.load(f)
    
    # List all video files in the directory
    available_videos = set(os.listdir(args.videos_dir))
    print(f"Found {len(available_videos)} video files in directory")
    
    # Process videos
    video_files = []
    found_count = 0
    not_found_count = 0
    
    for _, row in df.iterrows():
        # Use 'vid' instead of 'video_id'
        video_id = row['vid']
        
        # Find the video file
        video_path = find_video_file(args.videos_dir, video_id)
        
        if video_path:
            found_count += 1
            if not args.list_only:
                # Only pass the path and output directory, not the detector
                video_files.append((video_path, args.output_dir))
                
                # Limit the number of videos if specified
                if args.max_videos and found_count >= args.max_videos:
                    break
        else:
            not_found_count += 1
            print(f"Warning: Video file not found for ID: {video_id}")
    
    print(f"Found {found_count} videos, {not_found_count} videos not found")
    
    if args.list_only:
        print("List-only mode, exiting without processing videos")
        return
    
    if args.num_workers > 1:
        # Process videos in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(process_video_file, video_files))
        print(f"Processed {sum(results)} videos successfully out of {len(video_files)}")
    else:
        # Process videos sequentially
        results = []
        for video_path, output_dir in video_files:
            result = process_video_file((video_path, output_dir))
            results.append(result)
        print(f"Processed {sum(results)} videos successfully out of {len(video_files)}")

if __name__ == "__main__":
    main()
