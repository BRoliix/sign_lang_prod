# src/run_pipeline.py

import os
import sys
import argparse
import subprocess

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', '2D_to_3D'))

def process_data(data_subset="dev"):
    """Process raw OpenPose data through the Prompt2Sign pipeline"""
    print(f"Processing {data_subset} data...")
    
    # Step 1: JSON to H5
    subprocess.run([
        "python", 
        os.path.join("..", "tools", "2D_to_3D", "pipeline_demo_01_json2h5.py"), 
        "--data_subset", data_subset
    ])
    
    # Step 2: H5 to TXT
    subprocess.run([
        "python", 
        os.path.join("..", "tools", "2D_to_3D", "pipeline_demo_02_h5totxt.py"), 
        "--data_subset", data_subset
    ])
    
    # Step 3: TXT to SKELS
    subprocess.run([
        "python", 
        os.path.join("..", "tools", "2D_to_3D", "pipeline_demo_03_txt2skels.py"), 
        "--data_subset", data_subset
    ])
    
    print("Data processing complete!")

def main():
    parser = argparse.ArgumentParser(description="Sign Language Generation Pipeline")
    parser.add_argument("--mode", choices=["process", "emergency", "generate", "visualize"], required=True)
    parser.add_argument("--data_subset", choices=["dev", "train", "test"], default="dev")
    parser.add_argument("--text", type=str, help="Text to generate sign language for")
    parser.add_argument("--model_path", type=str, default="data/models/emergency_model.pt")
    parser.add_argument("--output_file", type=str, default="data/output/generated_pose.npy")
    parser.add_argument("--pose_file", type=str, help="Path to the pose file for visualization")
    parser.add_argument("--animation_file", type=str, default="data/output/animation.gif")
    
    args = parser.parse_args()
    
    if args.mode == "process":
        process_data(args.data_subset)
    elif args.mode == "emergency":
        from emergency_integration import emergency_setup
        emergency_setup()
    elif args.mode == "generate":
        if not args.text:
            parser.error("--text is required for generate mode")
        from src.generate_sign_language import generate_sign_language
        generate_sign_language(args.text, args.model_path, args.output_file)
    elif args.mode == "visualize":
        if not args.pose_file:
            parser.error("--pose_file is required for visualize mode")
        from src.visualize import create_animation
        import numpy as np
        try:
            import skeletalModel
            structure = skeletalModel.getSkeletalModelStructure()
        except:
            structure = [(i, i+1, i) for i in range(49)]
        pose_data = np.load(args.pose_file)
        create_animation(pose_data, args.animation_file, structure)
        print(f"Animation saved to {args.animation_file}")

if __name__ == "__main__":
    main()
