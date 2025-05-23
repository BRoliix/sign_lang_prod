# feature_extraction.py - Parallel processing version

import torch
import timm
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms
import argparse
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import time

class FeatureExtractor:
    def __init__(self, model_name='vit_small_patch14_dinov2.lvd142m'):
        # Disable fused attention to avoid errors on Mac M3
        timm.layers.set_fused_attn(False)
        
        # Load the pre-trained DINOv2 model
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        
        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features(self, image):
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transformation
        img_tensor = self.transform(image).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        return features.cpu().numpy()
    
    def extract_video_features(self, video_path, keypoints_path=None):
        # Improved video loading with error handling
        frames_features = []
        
        # Try multiple methods to open the video
        cap = self.safe_video_capture(video_path)
        
        if cap is None:
            print(f"Error: Could not open video {video_path}")
            return None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract features
            features = self.extract_features(frame)
            frames_features.append(features)
        
        cap.release()
        
        # Combine with keypoints if provided
        if keypoints_path and os.path.exists(keypoints_path):
            keypoints = np.load(keypoints_path)
            
            # Make sure we have the same number of frames
            min_frames = min(len(frames_features), len(keypoints))
            frames_features = frames_features[:min_frames]
            keypoints = keypoints[:min_frames]
            
            # Combine features and keypoints
            combined_features = []
            for i in range(min_frames):
                combined_features.append(np.concatenate([frames_features[i].flatten(), keypoints[i]]))
            
            return np.array(combined_features)
        
        return np.array(frames_features)
    
    def safe_video_capture(self, video_path):
        """Try multiple methods to open a video file."""
        # Method 1: Standard OpenCV VideoCapture
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            return cap
        
        # Method 2: Try with VideoCapture properties
        cap = cv2.VideoCapture()
        cap.open(video_path)
        if cap.isOpened():
            return cap
        
        # Method 3: Try with ffmpeg conversion
        try:
            import subprocess
            import tempfile
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Convert the video using ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-y', temp_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Try to open the converted video
            cap = cv2.VideoCapture(temp_path)
            if cap.isOpened():
                return cap
            
            # Clean up
            os.unlink(temp_path)
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}")
        
        return None

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
    
    # Try finding a file that contains the video_id
    for filename in os.listdir(videos_dir):
        if filename.endswith('.mp4') and video_id in filename:
            return os.path.join(videos_dir, filename)
    
    return None

def process_video(args):
    """Process a single video - designed for parallel execution"""
    video_data, args_dict = args
    
    # Unpack video data
    vid, video_path, keypoints_path, text = video_data
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    features_path = os.path.join(args_dict['output_dir'], f"{vid}_features.npy")
    text_path = os.path.join(args_dict['output_dir'], f"{vid}_text.txt")
    
    # Skip if features already exist
    if os.path.exists(features_path) and os.path.exists(text_path):
        return {'vid': vid, 'status': 'skipped'}
    
    try:
        # Extract features
        features = extractor.extract_video_features(video_path, keypoints_path)
        
        if features is not None:
            # Save features
            np.save(features_path, features)
            
            # Save text
            with open(text_path, 'w') as f:
                f.write(text)
            
            # Save processed video frames if requested
            if args_dict['save_processed_videos']:
                # Open the video again to save frames
                cap = extractor.safe_video_capture(video_path)
                if cap is not None:
                    frame_dir = os.path.join(args_dict['processed_videos_dir'], vid)
                    os.makedirs(frame_dir, exist_ok=True)
                    
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Save frame
                        frame_path = os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_count += 1
                    
                    cap.release()
            
            return {'vid': vid, 'status': 'success'}
        else:
            return {'vid': vid, 'status': 'failed', 'error': 'Could not extract features'}
    except Exception as e:
        return {'vid': vid, 'status': 'failed', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Extract features from videos')
    parser.add_argument('--tsv_path', type=str, default='data/raw/openasl/tsv/openasl-v1.0.tsv',
                        help='Path to OpenASL TSV file')
    parser.add_argument('--videos_dir', type=str, default='data/raw/openasl/videos',
                        help='Directory containing videos')
    parser.add_argument('--keypoints_dir', type=str, default='data/processed/keypoints',
                        help='Directory containing keypoints')
    parser.add_argument('--output_dir', type=str, default='data/processed/features',
                        help='Output directory for features')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process')
    parser.add_argument('--save_processed_videos', action='store_true',
                        help='Save processed video frames to disk')
    parser.add_argument('--processed_videos_dir', type=str, default='data/processed/videos',
                        help='Directory to save processed video frames')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (defaults to CPU count)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of videos to process in each batch')
    
    args = parser.parse_args()
    
    # Set default number of workers if not specified
    if args.num_workers is None:
        args.num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Load TSV file
    df = pd.read_csv(args.tsv_path, sep='\t', low_memory=False)
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Create output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.save_processed_videos and not os.path.exists(args.processed_videos_dir):
        os.makedirs(args.processed_videos_dir)
    
    # Prepare video data for processing
    video_data_list = []
    
    print(f"Preparing video data for processing...")
    for _, row in tqdm(df.iterrows(), total=len(df) if args.max_videos is None else min(len(df), args.max_videos)):
        # Use 'yid' for YouTube ID and 'vid' for the full video ID with timestamps
        yid = row['yid']
        vid = row['vid']
        text = row['raw-text']  # Use raw-text instead of text
        
        # Find the video file
        video_path = find_video_file(args.videos_dir, vid)
        
        # Skip if we've reached the maximum number of videos
        if args.max_videos and len(video_data_list) >= args.max_videos:
            break
        
        if video_path:
            # Check if keypoints exist
            keypoints_filename = f"{vid}_keypoints.npy"
            if vid.startswith('_-'):
                keypoints_filename = f"{vid[2:]}_keypoints.npy"
            keypoints_path = os.path.join(args.keypoints_dir, keypoints_filename)
            
            # Add to processing list
            video_data_list.append((vid, video_path, keypoints_path, text))
    
    # Process videos in parallel batches
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Create a dictionary of args for the worker function
    args_dict = {
        'output_dir': args.output_dir,
        'save_processed_videos': args.save_processed_videos,
        'processed_videos_dir': args.processed_videos_dir
    }
    
    # Process in batches
    total_batches = (len(video_data_list) + args.batch_size - 1) // args.batch_size
    
    print(f"Processing {len(video_data_list)} videos in {total_batches} batches with {args.num_workers} workers...")
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min((batch_idx + 1) * args.batch_size, len(video_data_list))
        batch = video_data_list[batch_start:batch_end]
        
        print(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch)} videos)...")
        
        # Process batch in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(process_video, [(video_data, args_dict) for video_data in batch]))
        
        # Count results
        for result in results:
            if result['status'] == 'success':
                processed_count += 1
                print(f"Processed {result['vid']}")
            elif result['status'] == 'skipped':
                skipped_count += 1
                print(f"Skipped {result['vid']} (already processed)")
            else:
                failed_count += 1
                print(f"Failed to process {result['vid']}: {result.get('error', 'Unknown error')}")
    
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds")
    print(f"Processed {processed_count} videos, skipped {skipped_count} videos, failed {failed_count} videos")
    
    if processed_count > 0:
        print(f"Average processing time per video: {elapsed_time/processed_count:.2f} seconds")

if __name__ == "__main__":
    main()
