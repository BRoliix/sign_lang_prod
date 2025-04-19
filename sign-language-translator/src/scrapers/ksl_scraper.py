# src/scrapers/ksl_scraper.py
import os
import requests
from bs4 import BeautifulSoup
import re
import json

class KSLScraper:
    def __init__(self, output_dir="data/raw/ksl"):
        self.base_url = "https://github.com/Yangseung/KSL"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def scrape(self):
        """Scrape the KSL dataset from GitHub"""
        print("Scraping KSL dataset...")
        
        # Get the main page
        response = requests.get(self.base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract repository information
        repo_info = {
            "name": "Korean Sign Language (KSL) Dataset",
            "url": self.base_url,
            "description": ""
        }
        
        # Extract description
        description_elem = soup.find("div", {"class": "markdown-body"})
        if description_elem:
            repo_info["description"] = description_elem.get_text()
        
        # Save repository information
        with open(os.path.join(self.output_dir, "dataset_info.json"), "w") as f:
            json.dump(repo_info, f, indent=4)
        
        # Find download links (raw files, releases, etc.)
        download_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and ('releases' in href or 'raw' in href or 'download' in href or 'archive' in href):
                full_url = href if href.startswith('http') else f"https://github.com{href}"
                download_links.append(full_url)
        
        # Save download links
        with open(os.path.join(self.output_dir, "download_links.txt"), "w") as f:
            for link in download_links:
                f.write(link + "\n")
        
        # Try to download the repository as ZIP
        try:
            zip_url = f"{self.base_url}/archive/refs/heads/main.zip"
            print(f"Downloading repository from {zip_url}...")
            response = requests.get(zip_url, stream=True)
            if response.status_code == 200:
                zip_path = os.path.join(self.output_dir, "ksl_repo.zip")
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("Repository downloaded successfully.")
            else:
                print(f"Failed to download repository: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading repository: {str(e)}")
        
        print("KSL dataset information saved.")
