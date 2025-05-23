# src/visualize.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from matplotlib.animation import FuncAnimation
import sys
from scipy.spatial.transform import Rotation as R

# Add the tools directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', '2D_to_3D'))

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
    
    # 3. Apply joint angle constraints
    # This is a simplified version - you might need more sophisticated constraints
    # For example, knees and elbows should only bend in one direction
    
    return normalized_pose

def visualize_pose_frame(ax, pose_frame, structure=None):
    """Visualize a single frame of pose data"""
    if structure is None:
        try:
            import importlib.util
            module_path = os.path.join(os.path.dirname(__file__), '..', 'tools', '2D_to_3D', 'skeletalModel.py')
            spec = importlib.util.spec_from_file_location("skeletalModel", module_path)
            skeletalModel = importlib.util.module_from_spec(spec)
            sys.modules["skeletalModel"] = skeletalModel
            spec.loader.exec_module(skeletalModel)
            structure = skeletalModel.getSkeletalModelStructure()
        except:
            # If skeletalModel is not available, create a simple structure
            structure = [(i, i+1, i) for i in range(min(49, pose_frame.shape[0]-1))]

    # Clear previous frame
    ax.clear()

    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot joints
    ax.scatter(pose_frame[:, 0], pose_frame[:, 1], pose_frame[:, 2], c='b', marker='o')

    # Plot connections
    for a, b, _ in structure:
        if a < pose_frame.shape[0] and b < pose_frame.shape[0]:
            ax.plot(
                [pose_frame[a, 0], pose_frame[b, 0]],
                [pose_frame[a, 1], pose_frame[b, 1]],
                [pose_frame[a, 2], pose_frame[b, 2]],
                'r-'
            )

    return ax

def create_animation(pose_data, output_file, structure=None):
    """Create an animation of the pose sequence"""
    # Normalize the pose data first
    normalized_pose_data = normalize_pose_data(pose_data)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        return visualize_pose_frame(ax, normalized_pose_data[frame_idx], structure)

    ani = FuncAnimation(fig, update, frames=min(100, len(normalized_pose_data)), interval=50)
    ani.save(output_file, writer='pillow', fps=20)
    
    # Also save a static image of the first frame
    plt.figure(figsize=(10, 10))
    static_ax = plt.subplot(111, projection='3d')
    visualize_pose_frame(static_ax, normalized_pose_data[0], structure)
    static_image = output_file.replace('.gif', '_first_frame.png')
    plt.savefig(static_image)
    
    return ani

def main():
    parser = argparse.ArgumentParser(description="Visualize generated sign language poses")
    parser.add_argument("--pose_file", type=str, required=True, help="Path to the pose file (.npy)")
    parser.add_argument("--output_file", type=str, default="pose_animation.gif", help="Path to save the animation")
    args = parser.parse_args()

    # Load pose data
    pose_data = np.load(args.pose_file)
    print(f"Loaded pose data with shape: {pose_data.shape}")

    # Get skeletal model structure
    try:
        import importlib.util
        module_path = os.path.join(os.path.dirname(__file__), '..', 'tools', '2D_to_3D', 'skeletalModel.py')
        spec = importlib.util.spec_from_file_location("skeletalModel", module_path)
        skeletalModel = importlib.util.module_from_spec(spec)
        sys.modules["skeletalModel"] = skeletalModel
        spec.loader.exec_module(skeletalModel)
        structure = skeletalModel.getSkeletalModelStructure()
    except Exception as e:
        print(f"Warning: Could not import skeletalModel. Using simple structure. ({e})")
        structure = [(i, i+1, i) for i in range(49)]

    # Create animation
    ani = create_animation(pose_data, args.output_file, structure)
    print(f"Animation saved to {args.output_file}")

if __name__ == "__main__":
    main()
