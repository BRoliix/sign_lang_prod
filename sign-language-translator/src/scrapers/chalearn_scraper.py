# src/scrapers/chalearn_scraper.py
import os
import requests
from bs4 import BeautifulSoup
import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

class ChalearnScraper:
    def __init__(self, output_dir="data/raw/chalearn"):
        self.base_url = "https://chalearnlap.cvc.uab.es/dataset/40/description/"
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
        """Scrape the ChaLearn dataset"""
        print("Scraping ChaLearn dataset...")
        
        # Navigate to the website
        self.driver.get(self.base_url)
        time.sleep(3)  # Wait for page to load
        
        # Extract dataset information
        page_content = self.driver.page_source
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Extract dataset description
        description = ""
        description_elem = soup.find("div", {"class": "dataset-description"})
        if description_elem:
            description = description_elem.get_text()
        else:
            # Try to get all text from the page
            description = soup.get_text()
        
        # Save dataset information
        with open(os.path.join(self.output_dir, "dataset_info.txt"), "w") as f:
            f.write(description)
        
        # Find download links
        download_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            text = link.get_text().lower()
            if href and ('download' in text or 'data' in text):
                full_url = href if href.startswith('http') else self.base_url + href
                download_links.append(full_url)
        
        # Save download links
        with open(os.path.join(self.output_dir, "download_links.txt"), "w") as f:
            for link in download_links:
                f.write(link + "\n")
        
        print("ChaLearn dataset information saved.")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()
