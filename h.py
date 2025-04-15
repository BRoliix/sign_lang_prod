import os
import json
import torch
import numpy as np
import ollama
from PIL import Image
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
    
    def _setup_embeddings(self):
        """Initialize Ollama embeddings with DeepSeek model"""
        print("Initializing DeepSeek embeddings...")
        try:
            # Pull the model if not already available
            ollama.pull(self.ollama_model)
            
            # Create embeddings function - FIXED: removed embed_batch_size parameter
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
        """Initialize video diffusion model"""
        print("Setting up video diffusion model...")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        
        try:
            self.diffusion = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
                variant="fp16" if device in ["cuda", "mps"] else None
            ).to(device)
            
            # Memory optimizations
            if device == "cuda":
                self.diffusion.enable_model_cpu_offload()
            self.diffusion.enable_attention_slicing()
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
    
    def find_reference_videos(self, contexts):
        """Find relevant video references from retrieved contexts"""
        videos_dir = os.path.join(self.data_dir, "videos")
        video_refs = []
        
        for context in contexts:
            metadata = context.get("metadata", {})
            if "videos" in metadata and isinstance(metadata["videos"], list):
                video_refs.extend(metadata["videos"])
        
        # Find actual video files
        video_files = []
        for video_ref in video_refs:
            for ext in [".mp4", ".avi", ".mov"]:
                video_path = os.path.join(videos_dir, f"{video_ref}{ext}")
                if os.path.exists(video_path):
                    video_files.append(video_path)
                    break
        
        if video_files:
            print(f"Found {len(video_files)} reference videos")
        else:
            print("No reference videos found")
        
        return video_files
    
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
    
    def generate_pose_image(self, prompt):
        """Generate a pose image based on the description (placeholder)"""
        # In a real implementation, this would use a specialized model for sign pose generation
        # For now, we'll create a simple colored image as placeholder
        
        # Create a placeholder image
        img = Image.new('RGB', (512, 512), color=(240, 240, 240))
        
        # Save as reference
        img_path = os.path.join(self.data_dir, "pose_frame.png")
        img.save(img_path)
        
        return img
    
    def generate_video(self, pose_image, fps=8, num_frames=16):
        """Generate sign language video using diffusion model"""
        print("Generating sign language video...")
        try:
            # Generate video frames
            video_output = self.diffusion(
                image=pose_image,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=127,  # Controls motion intensity
                noise_aug_strength=0.1
            ).frames[0]
            
            return video_output
        except Exception as e:
            print(f"Error generating video: {e}")
            raise
    
    def translate(self, text, output_file="output_video.mp4"):
        """End-to-end text to sign language video translation"""
        print(f"\n=== Translating: '{text}' ===")
        
        # 1. Retrieve relevant sign language context
        contexts = self.retrieve_context(text)
        
        # 2. Check for reference videos
        reference_videos = self.find_reference_videos(contexts)
        
        # 3. Generate enhanced prompt
        prompt = self.generate_sign_prompt(text, contexts)
        print(f"Generated prompt: {prompt[:200]}...")
        
        # 4. Generate initial pose image
        pose_image = self.generate_pose_image(prompt)
        
        # 5. Generate video
        video_frames = self.generate_video(pose_image)
        
        # 6. Save video
        output_path = os.path.join(self.data_dir, output_file)
        video_path = export_to_video(video_frames, output_path)
        print(f"Video saved to: {video_path}")
        
        return {
            "text": text,
            "prompt": prompt,
            "contexts": contexts,
            "reference_videos": reference_videos,
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
    
    args = parser.parse_args()
    
    # Initialize the system
    translator = SignLanguageRAG(
        data_dir=args.data,
        chroma_dir=args.db
    )
    
    # Translate text to sign language video
    result = translator.translate(args.text, args.output)
    
    print(f"\nTranslation complete!")
    print(f"Video: {result['video_path']}")
