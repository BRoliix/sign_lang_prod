# src/generate_sign_language.py

import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import SignLanguageRAG

def generate_sign_language(text, model_path="data/models/emergency_model.pt", output_file=None, pretrained_model_path=None):
    """Generate sign language from text using the trained model"""
    print(f"Generating sign language for: '{text}'")
    
    start_time = time.time()
    
    # Initialize RAG system with optional pretrained model path
    rag = SignLanguageRAG(model_path, pretrained_model_path)
    
    # Generate pose
    generated_pose = rag.generate(text)
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Print generation metrics
    print(f"\n{'='*50}")
    print(f"Generation Metrics:")
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Calculate pose statistics
    pose_mean = np.mean(generated_pose)
    pose_std = np.std(generated_pose)
    pose_min = np.min(generated_pose)
    pose_max = np.max(generated_pose)
    
    print(f"Pose statistics:")
    print(f" Mean: {pose_mean:.6f}")
    print(f" Std Dev: {pose_std:.6f}")
    print(f" Min: {pose_min:.6f}")
    print(f" Max: {pose_max:.6f}")
    print(f"{'='*50}\n")
    
    # Save output if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, generated_pose)
        print(f"Generated pose saved to {output_file}")
        
        # Create a simple visualization of the pose data
        plt.figure(figsize=(10, 6))
        if len(generated_pose.shape) == 3:
            plt.imshow(generated_pose[0], aspect='auto', cmap='viridis')
        else:
            plt.imshow(generated_pose, aspect='auto', cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(f'Generated Pose for: "{text}"')
        plt.xlabel('Frame')
        plt.ylabel('Feature')
        
        # Save the visualization
        viz_file = output_file.replace('.npy', '_viz.png')
        plt.savefig(viz_file)
        print(f"Pose visualization saved to {viz_file}")
    
    return generated_pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sign language from text")
    parser.add_argument("--text", type=str, required=True, help="Text to generate sign language for")
    parser.add_argument("--model_path", type=str, default="data/models/emergency_model.pt", help="Path to the model")
    parser.add_argument("--output_file", type=str, default="data/output/generated_pose.npy", help="Path to save the generated pose")
    parser.add_argument("--pretrained_model", type=str, help="Path to a pre-trained model (optional)")
    
    args = parser.parse_args()
    
    generate_sign_language(args.text, args.model_path, args.output_file, args.pretrained_model)
