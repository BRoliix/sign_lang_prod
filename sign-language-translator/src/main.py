# src/main.py

import argparse
import os
import sys
from tqdm import tqdm

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scrapers.how2sign_scraper import How2SignScraper
from src.scrapers.phoenix_scraper import PhoenixScraper
from src.scrapers.lsa64_scraper import LSA64Scraper
from src.scrapers.ksl_scraper import KSLScraper
from src.scrapers.chalearn_scraper import ChalearnScraper
from src.scrapers.openasl_scraper import OpenASLScraper  # Import the new scraper
from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.models.translator_model import SignLanguageTranslator
from src.utils.visualization import visualize_landmarks, visualize_sign_video

def main():
    parser = argparse.ArgumentParser(description='Sign Language Translator')
    parser.add_argument('--scrape', action='store_true', help='Scrape datasets')
    parser.add_argument('--dataset', type=str, 
                       choices=['all', 'how2sign', 'phoenix', 'lsa64', 'ksl', 'chalearn', 'openasl'],  # Add openasl option
                       default='all', help='Dataset to scrape')
    parser.add_argument('--process', action='store_true', help='Process videos')
    parser.add_argument('--extract', action='store_true', help='Extract landmarks')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--translate', type=str, help='Text to translate')
    parser.add_argument('--visualize', action='store_true', help='Visualize output')
    parser.add_argument('--input-dir', type=str, default='data/raw', help='Input directory for raw videos')
    parser.add_argument('--output-dir', type=str, default='data/processed/videos', help='Output directory for processed videos')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # Scrape datasets
    if args.scrape:
        print("Scraping datasets...")
        if args.dataset in ['all', 'how2sign']:
            scraper = How2SignScraper()
            scraper.scrape()
        
        if args.dataset in ['all', 'phoenix']:
            scraper = PhoenixScraper()
            scraper.scrape()
        
        if args.dataset in ['all', 'lsa64']:
            scraper = LSA64Scraper()
            scraper.scrape()
        
        if args.dataset in ['all', 'ksl']:
            scraper = KSLScraper()
            scraper.scrape()
        
        if args.dataset in ['all', 'chalearn']:
            scraper = ChalearnScraper()
            scraper.scrape()
            
        if args.dataset in ['all', 'openasl']:  # Add OpenASL scraping
            scraper = OpenASLScraper()
            scraper.scrape()
    
    # Process videos
    if args.process:
        print("Processing videos...")
        processor = VideoProcessor(input_dir=args.input_dir, output_dir=args.output_dir)
        processor.process_all_videos()
    
    # Extract landmarks
    if args.extract:
        print("Extracting landmarks...")
        extractor = LandmarkExtractor(input_dir=args.output_dir)
        extractor.extract_from_all_videos()
        extractor.create_dataset_csv()
    
    # Train model
    if args.train:
        print("Training model...")
        translator = SignLanguageTranslator()
        translator.train(epochs=50, batch_size=32)
    
    # Translate text
    if args.translate:
        print(f"Translating: '{args.translate}'")
        translator = SignLanguageTranslator(model_path='data/models/translator_model.h5')
        sign_sequence = translator.translate(args.translate)
        
        if args.visualize:
            visualize_sign_video(sign_sequence)
            visualize_landmarks(sign_sequence)

if __name__ == "__main__":
    main()
