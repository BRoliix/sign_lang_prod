# src/scrapers/phoenix_scraper.py
import os
import requests
from bs4 import BeautifulSoup
import re

class PhoenixScraper:
    def __init__(self, output_dir="data/raw/phoenix"):
        self.base_url = "https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def scrape(self):
        """Scrape the PHOENIX dataset"""
        print("Scraping PHOENIX dataset...")
        
        # Get the main page
        response = requests.get(self.base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract dataset information
        dataset_info = soup.get_text()
        
        # Save dataset information
        with open(os.path.join(self.output_dir, "dataset_info.txt"), "w") as f:
            f.write(dataset_info)
        
        # Find download links
        download_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and ('.zip' in href or '.tar.gz' in href or '.tgz' in href):
                full_url = href if href.startswith('http') else self.base_url + href
                download_links.append(full_url)
        
        # Save download links
        with open(os.path.join(self.output_dir, "download_links.txt"), "w") as f:
            for link in download_links:
                f.write(link + "\n")
        
        print("PHOENIX dataset information saved.")
        
        # Try to download a sample if available
        sample_links = [link for link in download_links if 'sample' in link.lower()]
        if sample_links:
            try:
                sample_url = sample_links[0]
                print(f"Downloading sample from {sample_url}...")
                response = requests.get(sample_url, stream=True)
                if response.status_code == 200:
                    sample_path = os.path.join(self.output_dir, "sample.zip")
                    with open(sample_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print("Sample downloaded successfully.")
            except Exception as e:
                print(f"Error downloading sample: {str(e)}")
