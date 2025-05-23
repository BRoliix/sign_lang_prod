# src/rag_system.py

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys 
import glob
from tqdm import tqdm

class SignLanguageEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print(f"Embedder initialized with model {model_name} on {self.device}")
        except Exception as e:
            print(f"Error initializing embedder: {e}")
            # Fallback to a simple embedding
            self.tokenizer = None
            self.model = None
            print("Using fallback embedding method")

    def get_embeddings(self, texts):
        if self.model is None or self.tokenizer is None:
            # Fallback: return random embeddings
            return np.random.randn(len(texts), 384)

        try:
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
                # Mean pooling
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return random embeddings as fallback
            return np.random.randn(len(texts), 384)

class PoseVectorDatabase:
    def __init__(self, features_path="data/processed/features"):
        self.features_path = features_path
        self.features = []
        self.filenames = []
        self.texts = []
        self.text_embeddings = None
        self.embedder = SignLanguageEmbedder()
        
    def load_data(self, texts_file="data/processed/texts.txt"):
        # Load features
        feature_files = glob.glob(os.path.join(self.features_path, "*.npy"))
        if not feature_files:
            print(f"No feature files found in {self.features_path}")
            return False

        print(f"Loading {len(feature_files)} feature files...")
        for file in tqdm(feature_files, desc="Loading features"):
            try:
                feature_data = np.load(file)
                self.features.append(feature_data)
                self.filenames.append(os.path.basename(file).replace('.npy', ''))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        # Load texts
        try:
            with open(texts_file, 'r') as f:
                self.texts = [line.strip() for line in f.readlines()]
            
            # If texts are fewer than features, pad with dummy texts
            if len(self.texts) < len(self.features):
                print(f"Warning: Only {len(self.texts)} texts for {len(self.features)} features. Creating dummy texts.")
                self.texts.extend([f"Sample text for feature {i}" for i in range(len(self.texts), len(self.features))])
        except Exception as e:
            print(f"Error loading texts: {e}. Creating dummy texts.")
            self.texts = [f"Sample text for {name}" for name in self.filenames]

        # Generate embeddings
        print("Generating text embeddings...")
        self.text_embeddings = self.embedder.get_embeddings(self.texts)
        
        print(f"Loaded {len(self.features)} features and {len(self.texts)} texts")
        return True
    
    def retrieve_similar_poses(self, query, top_k=5):
        # Get query embedding
        query_embedding = self.embedder.get_embeddings([query])[0]
        
        # Calculate similarity with all text embeddings
        similarities = cosine_similarity([query_embedding], self.text_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top-k pose data and their corresponding texts
        results = []
        for idx in top_indices:
            results.append({
                'feature': self.features[idx],
                'text': self.texts[idx],
                'similarity': similarities[idx]
            })
        return results

def preprocess_feature(feature_data):
    """Preprocess feature data to ensure it has the right shape for Conv2d"""
    feature = torch.tensor(feature_data, dtype=torch.float32)
    
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

class SignLanguageRAG:
    def __init__(self, model_path="data/models/emergency_model.pt", pretrained_model_path=None):
        self.vector_db = PoseVectorDatabase()
        self.vector_db.load_data()
        
        # Load diffusion model
        from src.diffusion_model import SignDiffusionModel
        
        # First try to load a pre-trained model if specified
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                # Load a pre-trained pose model that understands human pose structures
                print(f"Loading pre-trained model from {pretrained_model_path}")
                self.diffusion_model = torch.load(pretrained_model_path)
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                # Fall back to our model
                self.diffusion_model = SignDiffusionModel(fixed_channels=26)
                try:
                    self.diffusion_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    print(f"Loaded diffusion model from {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    print("Using untrained model")
        else:
            # Load our trained model
            self.diffusion_model = SignDiffusionModel(fixed_channels=26)
            try:
                self.diffusion_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Loaded diffusion model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Using untrained model")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()
        self.embedder = SignLanguageEmbedder()

    def generate(self, query, num_steps=50, guidance_scale=7.5):
        # Retrieve similar poses from the database
        similar_poses = self.vector_db.retrieve_similar_poses(query, top_k=3)
        
        # Print retrieved poses for debugging
        print(f"Retrieved {len(similar_poses)} similar poses:")
        for i, pose in enumerate(similar_poses):
            print(f"  {i+1}. Text: {pose['text'][:50]}... (Similarity: {pose['similarity']:.4f})")
        
        # Get text embedding for conditioning
        query_embedding = torch.tensor(
            self.embedder.get_embeddings([query])[0],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Get sample feature to determine shape
        if similar_poses:
            sample_feature = preprocess_feature(similar_poses[0]['feature']).to(self.device)
        else:
            # Default shape if no similar poses found
            sample_feature = torch.randn((1, 26, 1, 384), device=self.device)
        
        # Initialize from noise
        x = torch.randn_like(sample_feature)
        
        # Simple denoising loop
        for t in tqdm(range(1000, 0, -20), desc="Generating"):
            t_tensor = torch.tensor([t], device=self.device)
            with torch.no_grad():
                noise_pred = self.diffusion_model(x, t_tensor, query_embedding)
                x = x - 0.1 * noise_pred
        
        # Return the generated pose, squeezing if needed
        result = x.cpu().numpy()
        if result.shape[2] == 1:
            result = result.squeeze(2)
            
        return result[0]  # Return the first batch item
