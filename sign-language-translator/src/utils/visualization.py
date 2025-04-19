# src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
import os
from mpl_toolkits.mplot3d import Axes3D

def visualize_landmarks(landmarks, output_path=None):
    """Visualize landmarks as a 3D animation"""
    # MediaPipe hand connections
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Number of frames to visualize
    num_frames = min(len(landmarks), 100)
    frames_to_show = np.linspace(0, len(landmarks)-1, num_frames, dtype=int)
    
    # Visualize each frame
    for i, frame_idx in enumerate(frames_to_show):
        ax.clear()
        
        # Extract landmarks for current frame
        frame_landmarks = landmarks[frame_idx]
        
        # Reshape landmarks for left hand (first 21 landmarks)
        left_hand = frame_landmarks[:63].reshape(-1, 3)
        
        # Reshape landmarks for right hand (next 21 landmarks)
        right_hand = frame_landmarks[63:126].reshape(-1, 3)
        
        # Reshape landmarks for pose (remaining landmarks)
        pose = frame_landmarks[126:].reshape(-1, 3)
        
        # Plot left hand landmarks
        ax.scatter(left_hand[:, 0], left_hand[:, 1], left_hand[:, 2], c='r', marker='o')
        
        # Plot right hand landmarks
        ax.scatter(right_hand[:, 0], right_hand[:, 1], right_hand[:, 2], c='b', marker='o')
        
        # Plot pose landmarks
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='g', marker='s')
        
        # Connect landmarks with lines (simplified)
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if np.any(left_hand[start_idx]) and np.any(left_hand[end_idx]):
                ax.plot([left_hand[start_idx, 0], left_hand[end_idx, 0]],
                        [left_hand[start_idx, 1], left_hand[end_idx, 1]],
                        [left_hand[start_idx, 2], left_hand[end_idx, 2]], 'r-')
            
            if np.any(right_hand[start_idx]) and np.any(right_hand[end_idx]):
                ax.plot([right_hand[start_idx, 0], right_hand[end_idx, 0]],
                        [right_hand[start_idx, 1], right_hand[end_idx, 1]],
                        [right_hand[start_idx, 2], right_hand[end_idx, 2]], 'b-')
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx}')
        
        # Pause to create animation effect
        plt.pause(0.01)
        
        # Save frame if output path is provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, f'frame_{i:04d}.png'))
    
    plt.show()

def visualize_sign_video(landmarks, output_path=None, width=640, height=480):
    """Create a video visualization of the sign language landmarks"""
    # MediaPipe hand connections
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Create output video writer if path is provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Visualize each frame
    for frame_idx in range(len(landmarks)):
        # Create blank image
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Extract landmarks for current frame
        frame_landmarks = landmarks[frame_idx]
        
        # Reshape landmarks for left hand (first 21 landmarks)
        left_hand = frame_landmarks[:63].reshape(-1, 3)
        
        # Reshape landmarks for right hand (next 21 landmarks)
        right_hand = frame_landmarks[63:126].reshape(-1, 3)
        
        # Reshape landmarks for pose (remaining landmarks)
        pose = frame_landmarks[126:].reshape(-1, 3)
        
        # Create MediaPipe landmark objects
        left_hand_landmarks = mp.solutions.hands.HandLandmark
        right_hand_landmarks = mp.solutions.hands.HandLandmark
        
        # Draw left hand landmarks
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if np.any(left_hand[start_idx]) and np.any(left_hand[end_idx]):
                start_point = (int(left_hand[start_idx, 0] * width), int(left_hand[start_idx, 1] * height))
                end_point = (int(left_hand[end_idx, 0] * width), int(left_hand[end_idx, 1] * height))
                cv2.line(image, start_point, end_point, (0, 0, 255), 2)
        
        # Draw right hand landmarks
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if np.any(right_hand[start_idx]) and np.any(right_hand[end_idx]):
                start_point = (int(right_hand[start_idx, 0] * width), int(right_hand[start_idx, 1] * height))
                end_point = (int(right_hand[end_idx, 0] * width), int(right_hand[end_idx, 1] * height))
                cv2.line(image, start_point, end_point, (255, 0, 0), 2)
        
        # Draw pose landmarks
        for i in range(len(pose) - 1):
            if np.any(pose[i]) and np.any(pose[i+1]):
                start_point = (int(pose[i, 0] * width), int(pose[i, 1] * height))
                end_point = (int(pose[i+1, 0] * width), int(pose[i+1, 1] * height))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # Add frame number
        cv2.putText(image, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Display the image
        cv2.imshow('Sign Language Visualization', image)
        
        # Write to video if output path is provided
        if video_writer:
            video_writer.write(image)
        
        # Break on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release resources
    cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()
