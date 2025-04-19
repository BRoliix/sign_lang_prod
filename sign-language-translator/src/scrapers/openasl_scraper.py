# src/scrapers/openasl_scraper.py

import os
import requests
import subprocess
import json
from tqdm import tqdm

class OpenASLScraper:
    def __init__(self, output_dir="data/raw/openasl"):
        self.base_url = "https://github.com/chevalierNoir/OpenASL"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def scrape(self):
        """Scrape the OpenASL dataset"""
        print("Scraping OpenASL dataset...")
        
        # Create necessary directories
        os.makedirs(os.path.join(self.output_dir, "tsv"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "bbox"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        
        # Download the dataset TSV file
        tsv_url = "https://raw.githubusercontent.com/chevalierNoir/OpenASL/main/data/openasl-v1.0.tsv"
        tsv_path = os.path.join(self.output_dir, "tsv", "openasl-v1.0.tsv")
        self._download_file(tsv_url, tsv_path)
        
        # Download the bounding box file
        bbox_url = "https://raw.githubusercontent.com/chevalierNoir/OpenASL/main/data/bbox-v1.0.json"
        bbox_path = os.path.join(self.output_dir, "bbox", "bbox-v1.0.json")
        self._download_file(bbox_url, bbox_path)
        
        # Create a download script
        download_script = os.path.join(self.output_dir, "download_videos.py")
        self._create_download_script(download_script)
        
        print("OpenASL dataset information downloaded.")
        print("To download the videos, run:")
        print(f"python {download_script} --tsv {tsv_path} --dest {os.path.join(self.output_dir, 'videos')} --workers 2")
    
    def _download_file(self, url, output_path):
        """Download a file from URL"""
        print(f"Downloading {url} to {output_path}")
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {output_path}")
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
    
    def _create_download_script(self, script_path):
        """Create a Python script to download OpenASL videos"""
        script_content = '''
import argparse
import os
import subprocess
import pandas as pd
from tqdm import tqdm
import concurrent.futures

def convert_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS.mmm) to seconds"""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def download_video(video_id, start_time, end_time, output_dir):
    """Download a video segment using yt-dlp"""
    output_path = os.path.join(output_dir, f"{video_id}-{start_time}-{end_time}.mp4")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        return output_path
    
    # Format time for yt-dlp
    start_seconds = convert_time_to_seconds(start_time)
    end_seconds = convert_time_to_seconds(end_time)
    duration = end_seconds - start_seconds
    
    # Download the video segment
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--quiet",
        "--format", "best[height<=480]",
        "--output", output_path,
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg:-ss {start_seconds} -t {duration}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError:
        print(f"Failed to download {video_id} at {start_time}-{end_time}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download OpenASL videos')
    parser.add_argument('--tsv', type=str, required=True, help='Path to openasl-v1.0.tsv')
    parser.add_argument('--dest', type=str, required=True, help='Output directory for videos')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.dest, exist_ok=True)
    
    # Load the TSV file
    df = pd.read_csv(args.tsv, sep='\\t', low_memory=False)
    
    # Print column names to debug
    print("Available columns:", df.columns.tolist())
    
    # Find the correct column names
    video_id_col = None
    start_time_col = None
    end_time_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'video' in col_lower and 'id' in col_lower:
            video_id_col = col
        elif 'start' in col_lower and 'time' in col_lower:
            start_time_col = col
        elif 'end' in col_lower and 'time' in col_lower:
            end_time_col = col
    
    if not all([video_id_col, start_time_col, end_time_col]):
        print("Error: Could not find required columns in the TSV file.")
        print(f"Looking for video ID column (found: {video_id_col})")
        print(f"Looking for start time column (found: {start_time_col})")
        print(f"Looking for end time column (found: {end_time_col})")
        return
    
    print(f"Using columns: {video_id_col}, {start_time_col}, {end_time_col}")
    
    # Download videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for _, row in df.iterrows():
            video_id = str(row[video_id_col])
            start_time = str(row[start_time_col])
            end_time = str(row[end_time_col])
            futures.append(executor.submit(download_video, video_id, start_time, end_time, args.dest))
        
        # Show progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

if __name__ == '__main__':
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
