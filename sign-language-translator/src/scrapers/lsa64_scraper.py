# src/scrapers/lsa64_scraper.py
import os
import requests
from bs4 import BeautifulSoup
import re

class LSA64Scraper:
    def __init__(self, output_dir="data/raw/lsa64"):
        self.base_url = "https://facundoq.github.io/datasets/lsa64/"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def scrape(self):
        """Scrape the LSA64 dataset"""
        print("Scraping LSA64 dataset...")
        
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
            text = link.get_text().lower()
            if href and ('download' in text or '.zip' in href or '.tar.gz' in href):
                full_url = href if href.startswith('http') else self.base_url + href
                download_links.append(full_url)
        
        # Save download links
        with open(os.path.join(self.output_dir, "download_links.txt"), "w") as f:
            for link in download_links:
                f.write(link + "\n")
        
        print("LSA64 dataset information saved.")
        
        # Try to download the dataset if available
        if download_links:
            try:
                download_url = download_links[0]
                print(f"Downloading dataset from {download_url}...")
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    file_path = os.path.join(self.output_dir, "lsa64_dataset.zip")
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print("Dataset downloaded successfully.")
            except Exception as e:
                print(f"Error downloading dataset: {str(e)}")
