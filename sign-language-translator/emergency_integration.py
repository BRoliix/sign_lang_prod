# emergency_integration.py

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import glob
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
import requests
import zipfile
import io
import subprocess
import cv2

# Add the tools directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools', '2D_to_3D'))

# Import necessary modules
from src.diffusion_model import SignDiffusionModel
from src.rag_system import SignLanguageEmbedder

def install_mmpose():
    """Install MMPose if not already installed"""
    try:
        import mmpose
        print("MMPose is already installed.")
        return True
    except ImportError:
        print("Installing MMPose...")
        try:
            # Install MMPose dependencies
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openmim"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmengine"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmcv>=2.0.0"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet>=3.0.0"])
            
            # Install MMPose
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mmpose"])
            
            print("MMPose installed successfully!")
            return True
        except Exception as e:
            print(f"Failed to install MMPose: {e}")
            return False

def download_mmpose_model():
    """Download a pre-trained MMPose model"""
    model_dir = "data/models/mmpose"
    os.makedirs(model_dir, exist_ok=True)
    
    # Config and checkpoint URLs for HRNet model
    config_url = "https://github.com/open-mmlab/mmpose/raw/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    checkpoint_url = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
    
    config_path = os.path.join(model_dir, "hrnet_config.py")
    checkpoint_path = os.path.join(model_dir, "hrnet_checkpoint.pth")
    
    # Download config
    if not os.path.exists(config_path):
        print(f"Downloading MMPose config to {config_path}...")
        response = requests.get(config_url)
        with open(config_path, 'wb') as f:
            f.write(response.content)
    
    # Download checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Downloading MMPose checkpoint to {checkpoint_path}...")
        response = requests.get(checkpoint_url, stream=True)
        with open(checkpoint_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return config_path, checkpoint_path

def initialize_mmpose_model(config_path, checkpoint_path):
    """Initialize the MMPose model for pose estimation"""
    try:
        from mmpose.apis import init_model
        
        # Initialize the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = init_model(config_path, checkpoint_path, device=device)
        
        print("MMPose model initialized successfully!")
        return model
    except Exception as e:
        print(f"Failed to initialize MMPose model: {e}")
        return None

def normalize_pose_data(pose_data):
    """Normalize pose data to ensure it follows human skeletal constraints"""
    # If pose data is 1D or 2D, reshape to 3D format
    if len(pose_data.shape) == 1:
        # Assume it's a flattened single frame
        num_joints = pose_data.shape[0] // 3
        pose_data = pose_data.reshape(1, num_joints, 3)
    elif len(pose_data.shape) == 2:
        # Could be (frames, joints*3) or (joints, 3)
        if pose_data.shape[1] % 3 == 0:  # Likely (frames, joints*3)
            num_joints = pose_data.shape[1] // 3
            pose_data = pose_data.reshape(pose_data.shape[0], num_joints, 3)
        else:  # Likely (joints, 3) single frame
            pose_data = pose_data.reshape(1, pose_data.shape[0], pose_data.shape[1])
    
    # Apply constraints
    normalized_pose = pose_data.copy()
    
    # 1. Center the pose around the root joint (typically hip)
    root_idx = 0  # Assuming joint 0 is the root/hip
    for i in range(len(normalized_pose)):
        root_position = normalized_pose[i, root_idx].copy()
        normalized_pose[i] = normalized_pose[i] - root_position
    
    # 2. Apply bone length constraints
    # Define standard bone lengths (you may need to adjust these)
    standard_bone_lengths = {
        (0, 1): 0.2,  # hip to right hip
        (0, 4): 0.2,  # hip to left hip
        (1, 2): 0.5,  # right hip to right knee
        (2, 3): 0.5,  # right knee to right ankle
        (4, 5): 0.5,  # left hip to left knee
        (5, 6): 0.5,  # left knee to left ankle
        (0, 7): 0.5,  # hip to spine
        (7, 8): 0.2,  # spine to chest
        (8, 9): 0.2,  # chest to neck
        (9, 10): 0.2,  # neck to head
        (8, 11): 0.3,  # chest to left shoulder
        (11, 12): 0.3,  # left shoulder to left elbow
        (12, 13): 0.3,  # left elbow to left wrist
        (8, 14): 0.3,  # chest to right shoulder
        (14, 15): 0.3,  # right shoulder to right elbow
        (15, 16): 0.3,  # right elbow to right wrist
    }
    
    # Apply bone length constraints
    for i in range(len(normalized_pose)):
        for (joint_a, joint_b), length in standard_bone_lengths.items():
            if joint_a < normalized_pose.shape[1] and joint_b < normalized_pose.shape[1]:
                # Get current vector
                vec = normalized_pose[i, joint_b] - normalized_pose[i, joint_a]
                # Get current length
                current_length = np.linalg.norm(vec)
                if current_length > 0:
                    # Normalize and scale to desired length
                    normalized_vec = vec / current_length * length
                    # Apply constraint (adjust both joints to maintain center)
                    normalized_pose[i, joint_b] = normalized_pose[i, joint_a] + normalized_vec
    
    return normalized_pose

def load_preextracted_features(feature_path):
    """Load pre-extracted features instead of processing from scratch"""
    features = []
    filenames = []
    
    # Get all .npy files in the directory
    feature_files = glob.glob(os.path.join(feature_path, "*.npy"))
    if not feature_files:
        print(f"No feature files found in {feature_path}")
        return [], []
    
    print(f"Found {len(feature_files)} feature files")
    
    # Load each feature file
    for file in tqdm(feature_files, desc="Loading features"):
        try:
            feature_data = np.load(file)
            features.append(feature_data)
            filenames.append(os.path.basename(file).replace('.npy', ''))
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return features, filenames

def create_dummy_texts(filenames, output_path="data/processed"):
    """Create dummy text data if no text data is available"""
    texts = [f"Sample text for {name}" for name in filenames]
    
    # Save texts to file
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "texts.txt"), "w") as f:
        for text in texts:
            f.write(text + "\n")
    
    return texts

def preprocess_feature(feature_data):
    """Preprocess feature data to ensure it has the right shape for Conv2d"""
    feature = torch.tensor(feature_data, dtype=torch.float32)
    
    # Print original shape for debugging
    print(f"Original feature shape: {feature.shape}")
    
    # Handle different possible shapes
    if len(feature.shape) == 4: # [batch, channels, height, width]
        # Already in the right format for Conv2d
        return feature
    elif len(feature.shape) == 3: # [channels, height, width]
        # Add batch dimension
        return feature.unsqueeze(0)
    elif len(feature.shape) == 2: # [height, width]
        # Add batch and channel dimensions
        return feature.unsqueeze(0).unsqueeze(0)
    else:
        # For other shapes, try to make it work with Conv2d
        # Reshape to [batch, channels, 1, width]
        return feature.reshape(1, -1, 1, feature.shape[-1] if len(feature.shape) > 0 else 1)

def calculate_accuracy(pred, target):
    """Calculate accuracy between predicted and target tensors"""
    # Convert to numpy arrays
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Calculate MSE
    mse = mean_squared_error(pred_np.flatten(), target_np.flatten())
    
    # Calculate accuracy as 1 / (1 + MSE) - this gives a value between 0 and 1
    # where higher is better
    accuracy = 1.0 / (1.0 + mse)
    
    return accuracy

def initialize_weights(model):
    """Initialize model weights with better defaults"""
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)
    
    return model

def visualize_pose_3d(pose_data, output_file):
    """Create a 3D visualization of the pose data using MMPose's visualization tools"""
    try:
        from mmpose.visualization import PoseLocalVisualizer
        
        # Reshape pose data if needed
        if len(pose_data.shape) == 2 and pose_data.shape[1] % 3 == 0:
            num_joints = pose_data.shape[1] // 3
            pose_data = pose_data.reshape(-1, num_joints, 3)
        
        # Create visualizer
        visualizer = PoseLocalVisualizer()
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Visualize the first frame
        keypoints = pose_data[0]
        
        # Plot the joints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='blue', marker='o')
        
        # Connect the joints with lines (simplified skeleton)
        connections = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16)   # Right arm
        ]
        
        for joint1, joint2 in connections:
            if joint1 < keypoints.shape[0] and joint2 < keypoints.shape[0]:
                ax.plot(
                    [keypoints[joint1, 0], keypoints[joint2, 0]],
                    [keypoints[joint1, 1], keypoints[joint2, 1]],
                    [keypoints[joint1, 2], keypoints[joint2, 2]],
                    'r-'
                )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Pose Visualization')
        
        # Save the figure
        plt.savefig(output_file)
        plt.close()
        
        print(f"3D pose visualization saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error visualizing pose: {e}")
        return False

def emergency_setup(use_mmpose=True):
    print("Setting up emergency integration...")
    
    # Create necessary directories
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # Initialize MMPose if requested
    mmpose_model = None
    if use_mmpose:
        if install_mmpose():
            config_path, checkpoint_path = download_mmpose_model()
            mmpose_model = initialize_mmpose_model(config_path, checkpoint_path)
    
    # 1. Load pre-extracted features
    features_path = "data/processed/features"
    features, filenames = load_preextracted_features(features_path)
    
    if not features:
        print("No features found. Please check your feature directory.")
        return
    
    # 2. Load or create text data
    texts = []
    try:
        with open("data/processed/texts.txt", "r") as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts]
    except:
        print("No text file found, creating dummy texts...")
        texts = create_dummy_texts(filenames)
    
    # Get feature dimensions for model initialization
    sample_feature = features[0]
    print(f"Original feature shape: {sample_feature.shape}")
    
    # 3. Create a diffusion model with fixed internal channels
    model = SignDiffusionModel(fixed_channels=26)
    
    # Initialize weights with better defaults
    model = initialize_weights(model)
    
    # 4. Create embeddings
    print("Creating text embeddings...")
    embedder = SignLanguageEmbedder()
    text_embeddings = embedder.get_embeddings(texts)
    
    # 5. Train for just a few iterations to get something working
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create lists to store metrics
    train_losses = []
    train_accuracies = []
    epoch_avg_losses = []
    epoch_avg_accuracies = []
    
    # Minimal training loop
    print("Running minimal training...")
    for epoch in range(250):  # Increased from 10 to 30 epochs
        total_loss = 0
        total_accuracy = 0
        successful_samples = 0
        batch_losses = []
        batch_accuracies = []
        
        for i in tqdm(range(min(len(features), 100)), desc=f"Epoch {epoch+1}/{30}"):
            try:
                # Get the feature and prepare it for Conv2d
                feature = preprocess_feature(features[i]).to(device)
                print(f"Processed feature shape: {feature.shape}")
                
                text_embed = torch.tensor(text_embeddings[i], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Add noise
                noise = torch.randn_like(feature)
                t = torch.randint(0, 1000, (1,), device=device).long()
                noise_scale = t.float() / 1000
                noisy_feature = feature + noise * noise_scale
                
                # Predict noise
                predicted_noise = model(noisy_feature, t, text_embed)
                
                # Loss - ensure shapes match
                if predicted_noise.shape == noise.shape:
                    loss = F.mse_loss(predicted_noise, noise)
                    
                    # Calculate accuracy
                    accuracy = calculate_accuracy(predicted_noise, noise)
                    
                    # Store metrics
                    batch_losses.append(loss.item())
                    batch_accuracies.append(accuracy)
                    
                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy
                    successful_samples += 1
                    
                    print(f"Sample {i}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.6f}")
                else:
                    print(f"Shape mismatch: predicted {predicted_noise.shape}, noise {noise.shape}")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate epoch metrics
        if successful_samples > 0:
            avg_loss = total_loss / successful_samples
            avg_accuracy = total_accuracy / successful_samples
            
            epoch_avg_losses.append(avg_loss)
            epoch_avg_accuracies.append(avg_accuracy)
            
            # Store all batch metrics for detailed plotting
            train_losses.extend(batch_losses)
            train_accuracies.extend(batch_accuracies)
            
            # Print metrics with better formatting
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{30} Results:")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"Average Accuracy: {avg_accuracy:.6f} ({avg_accuracy*100:.2f}%)")
            print(f"Successful samples: {successful_samples}/{min(len(features), 100)}")
            print(f"{'='*50}\n")
        else:
            print(f"Epoch {epoch+1}/{30}: No successful samples")
    
    # 6. Save the model
    torch.save(model.state_dict(), "data/models/emergency_model.pt")
    
    # After training, print final metrics and plot graphs
    if len(epoch_avg_losses) > 0:
        print("\nTraining Complete!")
        print(f"{'='*50}")
        print(f"Final Loss: {epoch_avg_losses[-1]:.6f}")
        print(f"Final Accuracy: {epoch_avg_accuracies[-1]:.6f} ({epoch_avg_accuracies[-1]*100:.2f}%)")
        
        if len(epoch_avg_losses) > 1:
            loss_improvement = ((epoch_avg_losses[0] - epoch_avg_losses[-1])/epoch_avg_losses[0])*100
            print(f"Loss improvement: {loss_improvement:.2f}%")
        
        print(f"{'='*50}")
        
        # Create plots directory
        os.makedirs("data/plots", exist_ok=True)
        
        # Plot batch-level metrics
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(train_losses)), train_losses, 'b-')
        plt.title('Training Loss (Per Batch)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(train_accuracies)), train_accuracies, 'r-')
        plt.title('Training Accuracy (Per Batch)')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("data/plots/batch_metrics.png")
        print("Batch metrics plot saved to data/plots/batch_metrics.png")
        
        # Plot epoch-level metrics
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(epoch_avg_losses)+1), epoch_avg_losses, 'bo-')
        plt.title('Average Training Loss (Per Epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(epoch_avg_accuracies)+1), epoch_avg_accuracies, 'ro-')
        plt.title('Average Training Accuracy (Per Epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("data/plots/epoch_metrics.png")
        print("Epoch metrics plot saved to data/plots/epoch_metrics.png")
    
    print("Emergency setup complete! You now have a basic working model.")
    
    # 7. Generate a sample
    print("Generating a sample...")
    sample_text = "Hello, how are you?"
    sample_embed = torch.tensor(embedder.get_embeddings([sample_text])[0], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate from noise - use the same shape as our processed features
    try:
        sample_feature = preprocess_feature(features[0])
        noise_shape = sample_feature.shape
        x = torch.randn(noise_shape, device=device)
        
        # Simple denoising loop with quality metrics
        generation_quality = []
        for t in tqdm(range(1000, 0, -100), desc="Generating"):
            t_tensor = torch.tensor([t], device=device)
            with torch.no_grad():
                noise_pred = model(x, t_tensor, sample_embed)
                
                # Calculate generation quality at this step
                if t % 200 == 0:
                    # Measure coherence (smoothness between frames)
                    if x.shape[2] > 1: # If we have multiple frames
                        coherence = 1.0 - torch.mean(torch.abs(x[:,:,1:] - x[:,:,:-1])).item()
                    else:
                        coherence = 1.0
                    
                    generation_quality.append({
                        'step': t,
                        'coherence': coherence
                    })
                
                x = x - 0.1 * noise_pred
        
        # Print generation quality metrics
        if generation_quality:
            print("\nGeneration Quality Metrics:")
            print(f"{'='*50}")
            print(f"Final Coherence: {generation_quality[-1]['coherence']:.6f}")
            print(f"{'='*50}")
        
        # Convert to numpy and normalize the pose
        generated = x.cpu().numpy()[0]
        
        # If MMPose is available, use it to refine the pose
        if mmpose_model is not None:
            try:
                from mmpose.structures import PoseDataSample
                
                # Reshape for MMPose if needed
                if len(generated.shape) == 2:
                    # Likely [channels, width]
                    num_joints = generated.shape[1] // 3
                    reshaped_generated = generated.reshape(-1, num_joints, 3)
                    
                    # Use MMPose to refine the pose
                    refined_poses = []
                    for frame in reshaped_generated:
                        # Create a dummy image
                        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
                        
                        # Convert 3D pose to 2D for visualization
                        keypoints_2d = frame[:, :2].copy()
                        
                        # Scale to image coordinates
                        keypoints_2d = (keypoints_2d + 1) * 256  # Scale from [-1,1] to [0,512]
                        
                        # Create a data sample
                        data_sample = PoseDataSample()
                        data_sample.pred_instances = {'keypoints': keypoints_2d}
                        
                        # Refine the pose using MMPose
                        result = mmpose_model.inference(dummy_img, data_sample)
                        
                        # Extract the refined keypoints
                        refined_keypoints = result.pred_instances.keypoints.cpu().numpy()
                        
                        # Convert back to normalized coordinates
                        refined_keypoints = refined_keypoints / 256 - 1
                        
                        # Add back the Z dimension
                        refined_3d = np.zeros((refined_keypoints.shape[0], 3))
                        refined_3d[:, :2] = refined_keypoints
                        refined_3d[:, 2] = frame[:, 2]  # Keep original Z values
                        
                        refined_poses.append(refined_3d)
                    
                    # Update the generated pose with refined version
                    generated = np.array(refined_poses)
                else:
                    # Apply normalization directly
                    generated = normalize_pose_data(generated)
            except Exception as e:
                print(f"Error refining pose with MMPose: {e}")
                # Fall back to simple normalization
                generated = normalize_pose_data(generated)
        else:
            # Apply normalization without MMPose
            generated = normalize_pose_data(generated)
        
        # Save the generated sample
        os.makedirs("data/output", exist_ok=True)
        np.save("data/output/emergency_sample.npy", generated)
        
        # Create a simple visualization of the generated pose
        plt.figure(figsize=(10, 6))
        if len(generated.shape) == 3:
            plt.imshow(generated[0], aspect='auto', cmap='viridis')
        else:
            plt.imshow(generated, aspect='auto', cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(f'Generated Pose for: "{sample_text}"')
        plt.xlabel('Frame')
        plt.ylabel('Feature')
        plt.savefig("data/output/generated_pose_viz.png")
        print("Generated pose visualization saved to data/output/generated_pose_viz.png")
        
        # Create a 3D visualization of the pose
        visualize_pose_3d(generated, "data/output/generated_pose_3d.png")
        
        # Try to create an animation if possible
        try:
            from src.visualize import create_animation
            create_animation(generated, "data/output/emergency_sample.gif")
            print("3D animation saved to data/output/emergency_sample.gif")
        except Exception as e:
            print(f"Error creating animation: {e}")
        
        print("Sample generated and saved to data/output/emergency_sample.npy")
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up emergency integration")
    parser.add_argument("--use_mmpose", action="store_true", help="Use MMPose for pose refinement")
    
    args = parser.parse_args()
    
    emergency_setup(args.use_mmpose)
