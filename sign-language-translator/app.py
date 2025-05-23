# app.py

import streamlit as st
import numpy as np
import os
import sys
import tempfile
import time
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import subprocess

# Add the project directories to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from src.rag_system import SignLanguageRAG
from src.diffusion_model import SignDiffusionModel

# Set page configuration
st.set_page_config(
    page_title="Sign Language Video Generator",
    page_icon="👋",
    layout="wide"
)

def create_skeleton_video(pose_data, output_path, fps=10):
    """Create a video file from pose data using matplotlib and opencv"""
    # Create a temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Reshape pose data if needed
        if len(pose_data.shape) == 2:
            # Assuming pose_data is (frames, joints*3)
            num_frames = pose_data.shape[0]
            num_joints = pose_data.shape[1] // 3
            pose_data = pose_data.reshape(num_frames, num_joints, 3)
        
        # Create figure for animation
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Try to import the skeletal model structure if available
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'tools', '2D_to_3D'))
            from tools._2D_to_3D import skeletalModel
            structure = skeletalModel.getSkeletalModelStructure()
        except:
            # Create a simple structure for connections
            structure = [(i, i+1, i) for i in range(min(49, pose_data.shape[1]-1))]
        
        # Generate frames
        frame_files = []
        num_frames = min(100, len(pose_data))  # Limit to 100 frames for performance
        
        for frame_idx in range(num_frames):
            # Clear previous plot
            ax.clear()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Plot joints
            ax.scatter(
                pose_data[frame_idx, :, 0], 
                pose_data[frame_idx, :, 1], 
                pose_data[frame_idx, :, 2], 
                c='b', marker='o'
            )
            
            # Plot connections
            for a, b, _ in structure:
                if a < pose_data.shape[1] and b < pose_data.shape[1]:
                    ax.plot(
                        [pose_data[frame_idx, a, 0], pose_data[frame_idx, b, 0]],
                        [pose_data[frame_idx, a, 1], pose_data[frame_idx, b, 1]],
                        [pose_data[frame_idx, a, 2], pose_data[frame_idx, b, 2]],
                        'r-'
                    )
            
            # Save frame
            frame_file = os.path.join(temp_dir, f"frame_{frame_idx:03d}.png")
            plt.savefig(frame_file)
            frame_files.append(frame_file)
        
        plt.close(fig)
        
        # Create video using OpenCV instead of imageio
        if frame_files:
            # Read the first frame to get dimensions
            first_frame = cv2.imread(frame_files[0])
            height, width, layers = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Add frames to video
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                video.write(frame)
            
            # Release the video writer
            video.release()
            
            # Convert to browser-compatible format using ffmpeg
            mp4_temp = output_path + ".temp.mp4"
            os.rename(output_path, mp4_temp)
            
            try:
                # Use ffmpeg to convert to a web-compatible format
                subprocess.run([
                    'ffmpeg', '-y', '-i', mp4_temp, 
                    '-vcodec', 'libx264', 
                    '-pix_fmt', 'yuv420p', 
                    '-strict', '-2',
                    output_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Remove temporary file
                os.remove(mp4_temp)
            except Exception as e:
                st.error(f"Error converting video: {e}")
                # If conversion fails, use the original file
                os.rename(mp4_temp, output_path)
        
        return output_path

def main():
    # Create a title for the app
    st.title("Sign Language Video Generator 👋")
    
    # Add a description
    st.markdown("""
    This application generates sign language videos from text input using a diffusion model with RAG (Retrieval Augmented Generation).
    Enter your text below and click the button to generate a sign language video.
    """)
    
    # Create a text input field
    text_input = st.text_area("Enter text to convert to sign language:", height=100)
    
    # Create a button to generate the sign language
    if st.button("Generate Sign Language Video"):
        if text_input:
            # Create a temporary directory for the outputs
            with tempfile.TemporaryDirectory() as temp_dir:
                # Start timer
                start_time = time.time()
                
                # Generate sign language with progress indicator
                with st.spinner("Generating sign language..."):
                    # Initialize RAG system
                    rag = SignLanguageRAG()
                    
                    # Generate pose
                    generated_pose = rag.generate(text_input)
                
                # Calculate generation time
                generation_time = time.time() - start_time
                
                # Create video file
                with st.spinner("Creating video..."):
                    video_file = os.path.join(temp_dir, "sign_language.mp4")
                    create_skeleton_video(generated_pose, video_file)
                    
                    # Check if video was created successfully
                    if os.path.exists(video_file) and os.path.getsize(video_file) > 0:
                        # Display the video
                        st.video(video_file)
                        
                        # Display generation metrics
                        st.subheader("Generation Metrics")
                        st.write(f"Generation time: {generation_time:.2f} seconds")
                        
                        # Create a download button for the video
                        with open(video_file, "rb") as file:
                            st.download_button(
                                label="Download Video",
                                data=file,
                                file_name=f"sign_language_{int(time.time())}.mp4",
                                mime="video/mp4"
                            )
                        
                        # Save the generated pose to a file
                        pose_file = os.path.join(temp_dir, "generated_pose.npy")
                        np.save(pose_file, generated_pose)
                        
                        # Create a download button for the pose data
                        with open(pose_file, "rb") as file:
                            st.download_button(
                                label="Download Pose Data",
                                data=file,
                                file_name=f"pose_data_{int(time.time())}.npy",
                                mime="application/octet-stream"
                            )
                    else:
                        st.error("Failed to create video. Please check if ffmpeg is installed.")
                        st.info("You can install ffmpeg using: pip install imageio-ffmpeg")
        else:
            st.warning("Please enter some text to generate sign language.")
    
    # Add information about the model
    st.sidebar.title("About")
    st.sidebar.info("""
    This application uses a diffusion model with RAG (Retrieval Augmented Generation) to convert text to sign language.
    
    The model was trained on the OpenASL dataset and generates 3D pose sequences that represent sign language gestures.
    """)
    
    # Add system information
    st.sidebar.title("System Info")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.sidebar.text(f"Running on: {device}")
    
    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        st.sidebar.text("FFmpeg is installed")
    except:
        st.sidebar.warning("FFmpeg not found. Install with:\npip install imageio-ffmpeg")

if __name__ == "__main__":
    main()
