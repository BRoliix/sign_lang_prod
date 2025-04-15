import os
import json
import gc
import torch
import numpy as np
import ollama
from PIL import Image, ImageDraw
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

class SignLanguageRAG:
    def __init__(self, data_dir="./data", chroma_dir="./chroma_db"):
        """
        Initialize the Sign Language RAG system with DeepSeek for text processing
        and Stable Video Diffusion for video generation.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.chroma_dir = os.path.abspath(chroma_dir)
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.features_dir = os.path.join(self.data_dir, "features")
        
        # Make sure necessary dirs exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Clear memory
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Verify data directory structure
        self._verify_data_access()
        
        # Initialize Ollama with DeepSeek model
        self.ollama_model = "deepseek-r1"
        self._setup_embeddings()
        
        # Initialize Vector Store
        self._setup_vectorstore()
        
        # Initialize Video Diffusion Pipeline
        self._setup_diffusion()
    
    def _verify_data_access(self):
        """Ensure data directory is accessible and has expected structure"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check videos directory
        videos_dir = os.path.join(self.data_dir, "videos")
        if not os.path.exists(videos_dir):
            print(f"Videos directory not found. Creating: {videos_dir}")
            os.makedirs(videos_dir, exist_ok=True)
        
        # Check for JSON files
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if not json_files:
            print(f"Warning: No JSON files found in {self.data_dir}")
        else:
            print(f"Found {len(json_files)} JSON files: {', '.join(json_files)}")
        
        # Check for processed data
        if not os.path.exists(self.processed_dir):
            print(f"Processed directory not found. Creating: {self.processed_dir}")
            os.makedirs(self.processed_dir, exist_ok=True)
        
        # Check for feature data
        if not os.path.exists(self.features_dir):
            print(f"Features directory not found. Creating: {self.features_dir}")
            os.makedirs(self.features_dir, exist_ok=True)
    
    def _setup_embeddings(self):
        """Initialize Ollama embeddings with DeepSeek model"""
        print("Initializing DeepSeek embeddings...")
        try:
            # Pull the model if not already available
            ollama.pull(self.ollama_model)
            
            # Create embeddings function
            self.embeddings = OllamaEmbeddings(
                model=self.ollama_model
            )
        except Exception as e:
            print(f"Error initializing Ollama embeddings: {e}")
            raise
    
    def _setup_vectorstore(self):
        """Set up or load Chroma vector store with sign language data"""
        # Check if vector store already exists
        if os.path.exists(self.chroma_dir) and os.listdir(self.chroma_dir):
            print(f"Loading existing vector store from {self.chroma_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings
            )
            return
        
        # Create new vector store
        print("Creating new vector store from data files...")
        os.makedirs(self.chroma_dir, exist_ok=True)
        
        # Process all JSON files in data directory
        documents = []
        metadatas = []
        
        # Process ASL descriptions
        asl_file = os.path.join(self.data_dir, "asl_descriptions.json")
        if os.path.exists(asl_file):
            try:
                with open(asl_file, 'r') as f:
                    asl_data = json.load(f)
                    for sign, description in asl_data.items():
                        documents.append(f"Sign: {sign}, Description: {description}")
                        metadatas.append({
                            "source": "asl_descriptions.json",
                            "sign": sign,
                            "type": "description"
                        })
            except Exception as e:
                print(f"Error processing ASL descriptions: {e}")
        
        # Process NSLT files (sequence lengths)
        nslt_files = [f for f in os.listdir(self.data_dir) if f.startswith("nslt_") and f.endswith(".json")]
        for nslt_file in nslt_files:
            try:
                with open(os.path.join(self.data_dir, nslt_file), 'r') as f:
                    nslt_data = json.load(f)
                    for item in nslt_data:
                        if isinstance(item, dict) and "gloss" in item:
                            gloss = item["gloss"]
                            content = f"Gloss: {gloss}"
                            if "description" in item:
                                content += f", Description: {item['description']}"
                            
                            documents.append(content)
                            metadatas.append({
                                "source": nslt_file,
                                "gloss": gloss,
                                "type": "nslt",
                                "sequence_length": nslt_file.split("_")[1].split(".")[0]
                            })
            except Exception as e:
                print(f"Error processing {nslt_file}: {e}")
        
        # Process WLASL data
        wlasl_file = os.path.join(self.data_dir, "WLASL_v0.3.json")
        if os.path.exists(wlasl_file):
            try:
                with open(wlasl_file, 'r') as f:
                    wlasl_data = json.load(f)
                    for item in wlasl_data:
                        if "gloss" in item:
                            gloss = item["gloss"]
                            instances = item.get("instances", [])
                            
                            # Extract video references
                            videos = []
                            for instance in instances:
                                if "video" in instance:
                                    video_id = instance["video"].split("/")[-1]
                                    videos.append(video_id)
                            
                            content = f"Sign: {gloss}, Examples: {len(instances)}"
                            if videos:
                                content += f", Videos: {', '.join(videos[:3])}"
                            
                            documents.append(content)
                            metadatas.append({
                                "source": "WLASL_v0.3.json",
                                "sign": gloss,
                                "videos": videos,
                                "type": "wlasl"
                            })
            except Exception as e:
                print(f"Error processing WLASL data: {e}")
        
        # Add processed landmark data if available
        if os.path.exists(self.processed_dir):
            landmark_files = [f for f in os.listdir(self.processed_dir) if f.endswith('_landmarks.json')]
            for landmark_file in landmark_files:
                video_id = landmark_file.split('_landmarks.json')[0]
                
                # Find corresponding sign in WLASL data
                sign = None
                for meta in metadatas:
                    if "videos" in meta and video_id in meta.get("videos", []):
                        sign = meta.get("sign")
                        break
                
                if sign:
                    documents.append(f"Processed Sign: {sign}, Video: {video_id}, Type: Landmarks")
                    metadatas.append({
                        "source": landmark_file,
                        "sign": sign,
                        "video_id": video_id,
                        "landmarks_path": os.path.join(self.processed_dir, landmark_file),
                        "features_path": os.path.join(self.features_dir, f"{video_id}_features.npy"),
                        "visualization_path": os.path.join(self.processed_dir, f"{video_id}_landmarks.png"),
                        "type": "processed_landmarks"
                    })
        
        # Create vector store
        if documents:
            print(f"Creating vector store with {len(documents)} documents")
            self.vectorstore = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.chroma_dir
            )
            self.vectorstore.persist()
        else:
            print("Warning: No documents processed for vector store")
            self.vectorstore = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings
            )
    
    def _setup_diffusion(self):
        """Initialize video diffusion model with memory optimizations"""
        print("Setting up video diffusion model with memory optimizations...")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        
        try:
            # First load model on CPU to avoid OOM during initialization
            self.diffusion = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float32,  # Use full precision to avoid green videos
                variant=None,  # No fp16 variant
            )
            
            # Apply memory optimizations compatible with StableVideoDiffusionPipeline
            self.diffusion.enable_attention_slicing(slice_size=1)
            
            # Move to device
            if device == "cuda":
                self.diffusion = self.diffusion.to(device)
            elif device == "mps":
                self.diffusion = self.diffusion.to(device)
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"Error setting up diffusion model: {e}")
            raise
    
    def retrieve_context(self, query, k=3):
        """Retrieve relevant sign language context from vector store"""
        print(f"Retrieving context for: {query}")
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            contexts = []
            for doc in results:
                contexts.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                print(f"Found context: {doc.page_content[:100]}...")
            return contexts
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def find_reference_landmarks(self, contexts):
        """Find relevant landmark files from retrieved contexts"""
        landmark_paths = []
        feature_paths = []
        visualization_paths = []
        
        for context in contexts:
            metadata = context.get("metadata", {})
            
            # First check if this is a processed landmark context
            if metadata.get("type") == "processed_landmarks":
                if "landmarks_path" in metadata and os.path.exists(metadata["landmarks_path"]):
                    landmark_paths.append(metadata["landmarks_path"])
                
                if "features_path" in metadata and os.path.exists(metadata["features_path"]):
                    feature_paths.append(metadata["features_path"])
                
                if "visualization_path" in metadata and os.path.exists(metadata["visualization_path"]):
                    visualization_paths.append(metadata["visualization_path"])
            
            # Otherwise look for videos and check if they've been processed
            elif "videos" in metadata and isinstance(metadata["videos"], list):
                for video_id in metadata["videos"]:
                    landmarks_path = os.path.join(self.processed_dir, f"{video_id}_landmarks.json")
                    features_path = os.path.join(self.features_dir, f"{video_id}_features.npy")
                    vis_path = os.path.join(self.processed_dir, f"{video_id}_landmarks.png")
                    
                    if os.path.exists(landmarks_path):
                        landmark_paths.append(landmarks_path)
                    
                    if os.path.exists(features_path):
                        feature_paths.append(features_path)
                    
                    if os.path.exists(vis_path):
                        visualization_paths.append(vis_path)
        
        if landmark_paths:
            print(f"Found {len(landmark_paths)} processed landmark files")
        else:
            print("No processed landmarks found")
        
        return {
            "landmark_paths": landmark_paths,
            "feature_paths": feature_paths,
            "visualization_paths": visualization_paths
        }
    
    def generate_sign_prompt(self, query, contexts):
        """Generate enhanced prompt for sign language video generation"""
        # Extract context information
        context_info = []
        for ctx in contexts:
            content = ctx.get("content", "")
            if content:
                context_info.append(content)
        
        # Use DeepSeek to enhance the prompt
        enhanced_prompt = ollama.chat(model=self.ollama_model, messages=[
            {"role": "system", "content": "You are an expert in American Sign Language. Generate a detailed description for a video showing ASL signing."},
            {"role": "user", "content": f"""
            Create a description for a video showing a person signing in American Sign Language.
            
            Text to translate: "{query}"
            
            Sign language context information:
            {' '.join(context_info)}
            
            Focus on clear hand movements, proper ASL grammar, and natural transitions between signs.
            """}
        ])
        
        return enhanced_prompt["message"]["content"]
    
    def load_landmark_data(self, landmark_paths, max_landmarks=5):
        """Load landmark data from files"""
        all_landmarks = []
        
        # Limit number of landmarks to avoid memory issues
        landmark_paths = landmark_paths[:max_landmarks]
        
        for path in landmark_paths:
            try:
                with open(path, 'r') as f:
                    landmarks = json.load(f)
                    all_landmarks.append({
                        "path": path,
                        "data": landmarks
                    })
            except Exception as e:
                print(f"Error loading landmarks from {path}: {e}")
        
        return all_landmarks
    
    def generate_pose_image_from_landmarks(self, landmarks_data, prompt, size=(512, 512)):
        """Generate a pose image based on landmark data"""
        # Create a blank image
        img = Image.new('RGB', size, color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Determine which frame to use (middle frame)
        if landmarks_data and landmarks_data[0]["data"]:
            landmarks = landmarks_data[0]["data"]
            frame_idx = len(landmarks) // 2
            if frame_idx < len(landmarks):
                frame = landmarks[frame_idx]
                
                # Scale factor to fit landmarks in the image
                scale_factor = min(size) * 0.8
                center_x, center_y = size[0] // 2, size[1] // 2
                
                # Draw pose landmarks
                if frame.get("pose"):
                    self._draw_landmarks(draw, frame["pose"], center_x, center_y, scale_factor, 
                                         color=(255, 0, 0), size=4)
                
                # Draw hand landmarks (most important for sign language)
                if frame.get("right_hand"):
                    self._draw_landmarks(draw, frame["right_hand"], center_x, center_y, scale_factor,
                                        color=(0, 255, 0), size=6)
                
                if frame.get("left_hand"):
                    self._draw_landmarks(draw, frame["left_hand"], center_x, center_y, scale_factor,
                                        color=(0, 0, 255), size=6)
                
                # Draw face landmarks
                if frame.get("face"):
                    self._draw_landmarks(draw, frame["face"], center_x, center_y, scale_factor,
                                        color=(255, 255, 0), size=2)
                
                # Add text based on the prompt
                lines = self._get_key_phrases(prompt)
                font_size = 12
                y_position = 10
                for line in lines:
                    draw.text((10, y_position), line, fill=(0, 0, 0))
                    y_position += font_size + 4
        
        # If we don't have landmark data, create a more colorful gradient
        else:
            # Create a gradient background
            for y in range(size[1]):
                r = int(240 - (y / size[1]) * 40)
                g = int(240 - (y / size[1]) * 30)
                b = int(240 - (y / size[1]) * 20)
                for x in range(size[0]):
                    draw.point((x, y), fill=(r, g, b))
            
            # Add some graphic elements that represent sign language
            # Draw hand outlines
            draw.ellipse((150, 150, 250, 250), outline=(0, 100, 0), width=3)
            draw.ellipse((250, 150, 350, 250), outline=(0, 0, 100), width=3)
            
            # Draw lines representing fingers
            for i in range(5):
                angle = i * 3.14159 / 5
                x1, y1 = 200, 200
                x2 = x1 + int(60 * np.cos(angle))
                y2 = y1 + int(60 * np.sin(angle))
                draw.line((x1, y1, x2, y2), fill=(0, 200, 0), width=2)
                
                x1, y1 = 300, 200
                x2 = x1 + int(60 * np.cos(angle))
                y2 = y1 + int(60 * np.sin(angle))
                draw.line((x1, y1, x2, y2), fill=(0, 0, 200), width=2)
        
        # Save as reference
        img_path = os.path.join(self.data_dir, "pose_frame.png")
        img.save(img_path)
        
        return img
    
    def _draw_landmarks(self, draw, landmarks, center_x, center_y, scale, color=(255, 0, 0), size=3):
        """Draw landmarks on the image"""
        # Draw points
        for point in landmarks:
            x = center_x + point["x"] * scale
            y = center_y + point["y"] * scale
            draw.ellipse((x-size, y-size, x+size, y+size), fill=color)
    
    def _get_key_phrases(self, prompt, max_phrases=5):
        """Extract key phrases from the prompt"""
        sentences = prompt.split('.')
        phrases = []
        
        for sentence in sentences:
            if len(phrases) >= max_phrases:
                break
            
            words = sentence.strip().split()
            if len(words) > 3:
                phrases.append(' '.join(words[:min(len(words), 7)]) + '...')
        
        return phrases[:max_phrases]
    
    def generate_video(self, pose_image, fps=8, num_frames=8, motion_bucket_id=127, noise_aug_strength=0.1):
        """Generate sign language video using diffusion model with reduced memory usage"""
        print("Generating sign language video with optimized memory settings...")
        try:
            # Clear memory cache for MPS
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Generate video frames with reduced parameters
            video_output = self.diffusion(
                image=pose_image,
                num_frames=num_frames,  # Reduced from default
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                num_inference_steps=20,  # Reduced steps for less memory
                height=256,  # Ensure smaller height
                width=256,   # Ensure smaller width
            ).frames[0]
            
            return video_output
        except RuntimeError as e:
            if "buffer size" in str(e):
                # Attempt with even more reduced parameters
                print("Memory error encountered. Retrying with reduced parameters...")
                return self._retry_with_reduced_params(pose_image, fps, num_frames)
            else:
                print(f"Error generating video: {e}")
                raise
    
    def _retry_with_reduced_params(self, pose_image, fps, num_frames):
        """Retry video generation with drastically reduced parameters"""
        # Resize image to even smaller dimensions
        small_image = pose_image.resize((192, 192), Image.LANCZOS)
        
        # Generate with minimal parameters
        video_output = self.diffusion(
            image=small_image,
            num_frames=min(6, num_frames),  # Very few frames
            fps=fps,
            motion_bucket_id=50,  # Lower motion intensity
            noise_aug_strength=0.05,
            num_inference_steps=15,  # Minimum steps
            height=192,
            width=192,
        ).frames[0]
        
        return video_output
    
    def translate(self, text, output_file="output_video.mp4", fps=8, num_frames=8, 
                 motion_bucket_id=127, noise_aug_strength=0.1):
        """End-to-end text to sign language video translation with memory optimizations"""
        print(f"\n=== Translating: '{text}' ===")
        
        # 1. Retrieve relevant sign language context
        contexts = self.retrieve_context(text)
        
        # 2. Check for reference landmark data
        landmark_refs = self.find_reference_landmarks(contexts)
        
        # 3. Load landmark data
        landmark_data = self.load_landmark_data(landmark_refs["landmark_paths"])
        
        # 4. Generate enhanced prompt
        prompt = self.generate_sign_prompt(text, contexts)
        print(f"Generated prompt: {prompt[:200]}...")
        
        # 5. Generate initial pose image from landmarks
        pose_image = self.generate_pose_image_from_landmarks(landmark_data, prompt)
        
        # 6. Generate video with memory-optimized parameters
        video_frames = self.generate_video(
            pose_image, 
            fps=fps, 
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength
        )
        
        # 7. Save video
        output_path = os.path.join(self.data_dir, output_file)
        video_path = export_to_video(video_frames, output_path, fps=fps)
        print(f"Video saved to: {video_path}")
        
        return {
            "text": text,
            "prompt": prompt,
            "contexts": contexts,
            "landmark_refs": landmark_refs,
            "visualization_paths": landmark_refs["visualization_paths"],
            "video_path": video_path
        }


# Client code to use the system
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text to Sign Language Video Translator")
    parser.add_argument("--text", default="Hello, how are you?", help="Text to translate")
    parser.add_argument("--data", default="./data", help="Data directory path")
    parser.add_argument("--db", default="./chroma_db", help="Vector store directory path")
    parser.add_argument("--output", default="output_video.mp4", help="Output video filename")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    
    args = parser.parse_args()
    
    # Initialize the system
    translator = SignLanguageRAG(
        data_dir=args.data,
        chroma_dir=args.db
    )
    
    # Translate text to sign language video
    result = translator.translate(
        args.text, 
        args.output, 
        fps=args.fps, 
        num_frames=args.frames
    )
    
    print(f"\nTranslation complete!")
    print(f"Video: {result['video_path']}")
