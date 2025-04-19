# src/scrapers/how2sign_scraper.py
import os
import requests
from bs4 import BeautifulSoup
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

class How2SignScraper:
    def __init__(self, output_dir="data/raw/how2sign"):
        self.base_url = "https://how2sign.github.io"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup Chrome options for Selenium
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Initialize the Chrome WebDriver
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
    
    def scrape(self):
        """Scrape the How2Sign dataset"""
        print("Scraping How2Sign dataset...")
        
        # Navigate to the website
        self.driver.get(self.base_url)
        time.sleep(3)  # Wait for page to load
        
        # Extract dataset information
        page_content = self.driver.page_source
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Extract dataset description
        description = ""
        for p in soup.find_all('p'):
            description += p.get_text() + "\n"
        
        # Save dataset information
        with open(os.path.join(self.output_dir, "dataset_info.txt"), "w") as f:
            f.write(description)
        
        # Find download links
        download_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and ('.zip' in href or '.tar.gz' in href or 'download' in href.lower()):
                full_url = href if href.startswith('http') else self.base_url + href
                download_links.append(full_url)
        
        # Save download links
        with open(os.path.join(self.output_dir, "download_links.txt"), "w") as f:
            for link in download_links:
                f.write(link + "\n")
        
        # Extract sample videos if available
        sample_videos = []
        for video in soup.find_all('video'):
            src = video.get('src')
            if src:
                full_url = src if src.startswith('http') else self.base_url + src
                sample_videos.append(full_url)
        
        # Download sample videos
        for i, video_url in enumerate(sample_videos):
            try:
                response = requests.get(video_url, stream=True)
                if response.status_code == 200:
                    video_path = os.path.join(self.output_dir, f"sample_video_{i+1}.mp4")
                    with open(video_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"Downloaded sample video {i+1}")
            except Exception as e:
                print(f"Error downloading video {i+1}: {str(e)}")
        
        print("How2Sign dataset information and samples saved.")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
