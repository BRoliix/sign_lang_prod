# evaluate_model.py

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from src.diffusion_model import SignDiffusionModel
from src.rag_system import SignLanguageRAG, SignLanguageEmbedder

def generate_poses(model, texts, device="cpu"):
    """Generate sign language poses for a list of texts"""
    embedder = SignLanguageEmbedder()
    rag = SignLanguageRAG(model_path=None)  # Use the provided model instead
    rag.diffusion_model = model
    
    generated_poses = []
    for text in tqdm(texts, desc="Generating poses"):
        pose = rag.generate(text)
        generated_poses.append(pose)
    
    return generated_poses

def evaluate_model(model_path, test_data_path, output_dir="evaluation_results"):
    """Evaluate a trained diffusion model"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = SignDiffusionModel(fixed_channels=26).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    test_features = []
    test_texts = []
    
    # Load features
    for file in os.listdir(os.path.join(test_data_path, "features")):
        if file.endswith('.npy'):
            test_features.append(np.load(os.path.join(test_data_path, "features", file)))
    
    # Load texts
    with open(os.path.join(test_data_path, "texts.txt"), 'r') as f:
        test_texts = [line.strip() for line in f.readlines()]
    
    # Generate poses
    generated_poses = generate_poses(model, test_texts, device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = {}
    
    # 1. Back Translation Metric (if SLT model is available)
    try:
        from slt_model import SLTModel  # Placeholder for your SLT model
        slt_model = SLTModel()
        bt_metrics = evaluate_back_translation(generated_poses, test_texts, slt_model)
        metrics.update(bt_metrics)
        print(f"Back Translation Metrics: BLEU = {bt_metrics['bleu']:.4f}, ROUGE-L = {bt_metrics['rouge-l']:.4f}")
    except ImportError:
        print("SLT model not available, skipping back translation metrics")
    
    # 2. FID Score
    try:
        real_features = np.array([feat.flatten() for feat in test_features])
        gen_features = np.array([feat.flatten() for feat in generated_poses])
        fid = calculate_fid(real_features, gen_features)
        metrics['fid'] = fid
        print(f"FID Score: {fid:.4f}")
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    # 3. Diversity Score
    try:
        diversity = calculate_diversity(generated_poses)
        metrics['diversity'] = diversity
        print(f"Diversity Score: {diversity:.4f}")
    except Exception as e:
        print(f"Error calculating diversity: {e}")
    
    # 4. Multimodal Distance
    try:
        # Get text embeddings
        embedder = SignLanguageEmbedder()
        text_embeddings = embedder.get_embeddings(test_texts)
        
        # Calculate multimodal distance
        mm_distance = calculate_multimodal_distance(text_embeddings, generated_poses, None)  # Replace None with your feature extractor
        metrics['multimodal_distance'] = mm_distance
        print(f"Multimodal Distance: {mm_distance:.4f}")
    except Exception as e:
        print(f"Error calculating multimodal distance: {e}")
    
    # Save metrics
    np.save(os.path.join(output_dir, "metrics.npy"), metrics)
    
    # Save example generations
    for i, (text, pose) in enumerate(zip(test_texts[:5], generated_poses[:5])):
        np.save(os.path.join(output_dir, f"example_{i}_pose.npy"), pose)
        with open(os.path.join(output_dir, f"example_{i}_text.txt"), 'w') as f:
            f.write(text)
    
    print(f"Evaluation results saved to {output_dir}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sign language generation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data, args.output_dir)
