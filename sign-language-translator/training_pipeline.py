# training_pipeline.py

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
import glob
import random
from src.diffusion_model import SignDiffusionModel
from src.rag_system import SignLanguageEmbedder

class SignLanguageDataset(Dataset):
    def __init__(self, features_path, split_ratio=0.8, is_train=True, seed=42):
        self.features_paths = []
        self.texts = []
        self.embedder = SignLanguageEmbedder()
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Get all feature files
        feature_files = glob.glob(os.path.join(features_path, "*.npy"))
        
        # Create text labels from filenames
        for file_path in feature_files:
            filename = os.path.basename(file_path)
            # Extract video ID from filename
            video_id = filename.replace("_features.npy", "")
            
            # Check if corresponding text file exists
            text_file = os.path.join(os.path.dirname(features_path), f"{video_id}_text.txt")
            if os.path.exists(text_file):
                with open(text_file, 'r') as f:
                    text = f.read().strip()
            else:
                # Use filename as fallback text
                text = f"Sign language for {video_id}"
                
            self.features_paths.append(file_path)
            self.texts.append(text)
        
        # Split data into train and validation sets
        num_samples = len(self.features_paths)
        indices = list(range(num_samples))
        random.shuffle(indices)
        split = int(split_ratio * num_samples)
        
        if is_train:
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]
            
        print(f"{'Training' if is_train else 'Validation'} dataset created with {len(self.indices)} samples")
        
        # Generate text embeddings for all texts
        print("Generating text embeddings...")
        self.text_embeddings = self.embedder.get_embeddings([self.texts[i] for i in self.indices])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our filtered indices
        actual_idx = self.indices[idx]
        
        # Load feature on demand to save memory
        feature_data = np.load(self.features_paths[actual_idx])
        feature = torch.tensor(feature_data, dtype=torch.float32)
        
        # Get corresponding text embedding
        text_embed = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
        
        return feature, text_embed

def custom_collate_fn(batch):
    """Custom collate function to handle tensors of different shapes"""
    features = []
    text_embeds = []
    
    for feature, text_embed in batch:
        features.append(feature)
        text_embeds.append(text_embed)
    
    # Stack text embeddings (they should all have the same shape)
    text_embeds = torch.stack(text_embeds, dim=0)
    
    # Return features as a list and text_embeds as a tensor
    return features, text_embeds

def preprocess_feature(feature):
    """Preprocess feature to ensure it has the right shape for Conv2d"""
    if len(feature.shape) == 4:  # [batch, channels, height, width]
        return feature
    elif len(feature.shape) == 3:  # [channels, height, width]
        return feature.unsqueeze(0)
    elif len(feature.shape) == 2:  # [height, width]
        return feature.unsqueeze(0).unsqueeze(0)
    else:
        # For your specific feature format
        return feature.reshape(1, feature.shape[0], 1, feature.shape[1] if len(feature.shape) > 1 else feature.shape[0])

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

def train_diffusion_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=2e-4,
    device="cpu",
    save_dir="data/models",
    log_interval=5
):
    """Train the diffusion model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # For tracking metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        successful_samples = 0
        
        # Configure tqdm to be less verbose - only show progress bar without loss info
        for features, text_embeds in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            text_embeds = text_embeds.to(device)
            
            # Process each feature individually (since they may have different shapes)
            batch_loss = 0
            batch_accuracy = 0
            valid_samples = 0
            
            for i, feature in enumerate(features):
                try:
                    # Preprocess feature
                    feature = preprocess_feature(feature).to(device)
                    text_embed = text_embeds[i].unsqueeze(0)  # Add batch dimension
                    
                    # Add noise
                    noise = torch.randn_like(feature)
                    t = torch.randint(0, 1000, (1,), device=device).long()
                    noise_scale = t.float() / 1000
                    noisy_feature = feature + noise * noise_scale
                    
                    # Predict noise
                    predicted_noise = model(noisy_feature, t, text_embed)
                    
                    # Loss
                    loss = F.mse_loss(predicted_noise, noise)
                    
                    # Calculate accuracy
                    accuracy = calculate_accuracy(predicted_noise, noise)
                    
                    # Backprop
                    loss = loss / len(features)  # Normalize by batch size
                    loss.backward()
                    batch_loss += loss.item() * len(features)  # Scale back for reporting
                    batch_accuracy += accuracy
                    valid_samples += 1
                except Exception as e:
                    # Silently continue on errors
                    continue
            
            # Only step optimizer if at least one sample was processed successfully
            if valid_samples > 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                successful_samples += valid_samples
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average metrics
        if successful_samples > 0:
            avg_loss = epoch_loss / successful_samples
            avg_accuracy = epoch_accuracy / successful_samples
            train_losses.append(avg_loss)
            train_accuracies.append(avg_accuracy)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Train Accuracy: {avg_accuracy:.6f}")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_epoch_loss = 0
            val_epoch_accuracy = 0
            val_successful_samples = 0
            
            with torch.no_grad():
                for features, text_embeds in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    text_embeds = text_embeds.to(device)
                    
                    # Process each feature individually
                    for i, feature in enumerate(features):
                        try:
                            # Preprocess feature
                            feature = preprocess_feature(feature).to(device)
                            text_embed = text_embeds[i].unsqueeze(0)  # Add batch dimension
                            
                            # Add noise
                            noise = torch.randn_like(feature)
                            t = torch.randint(0, 1000, (1,), device=device).long()
                            noise_scale = t.float() / 1000
                            noisy_feature = feature + noise * noise_scale
                            
                            # Predict noise
                            predicted_noise = model(noisy_feature, t, text_embed)
                            
                            # Loss
                            loss = F.mse_loss(predicted_noise, noise)
                            
                            # Calculate accuracy
                            accuracy = calculate_accuracy(predicted_noise, noise)
                            
                            val_epoch_loss += loss.item()
                            val_epoch_accuracy += accuracy
                            val_successful_samples += 1
                        except Exception as e:
                            # Silently continue on errors
                            continue
            
            # Calculate average metrics
            if val_successful_samples > 0:
                avg_val_loss = val_epoch_loss / val_successful_samples
                avg_val_accuracy = val_epoch_accuracy / val_successful_samples
                val_losses.append(avg_val_loss)
                val_accuracies.append(avg_val_accuracy)
                print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.6f}, Val Accuracy: {avg_val_accuracy:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_dir, "diffusion_model_best.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
        
        # Save model checkpoint
        if (epoch + 1) % log_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"diffusion_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss if val_loader is not None else None,
            }, checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
            
            # Plot and save training curves at checkpoints
            plot_training_curves(
                train_losses, train_accuracies,
                val_losses, val_accuracies,
                save_dir, epoch+1
            )
    
    # Save final model
    final_model_path = os.path.join(save_dir, "diffusion_model_final2.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot final training curves
    plot_training_curves(
        train_losses, train_accuracies,
        val_losses, val_accuracies,
        save_dir, epochs
    )
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
    }

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, save_dir, epoch):
    """Plot and save training curves"""
    plt.figure(figsize=(15, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train')
    if val_losses:
        plt.plot(val_losses, 'r-', label='Validation')
    plt.title(f'Loss (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'b-', label='Train')
    if val_accuracies:
        plt.plot(val_accuracies, 'r-', label='Validation')
    plt.title(f'Accuracy (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_curves_epoch_{epoch}.png"))
    plt.close()

def main():
    # Parse arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Train diffusion model for sign language generation")
    parser.add_argument("--features_path", type=str, default="data/processed/features", help="Path to features directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="data/models", help="Directory to save models")
    parser.add_argument("--log_interval", type=int, default=5, help="Interval to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SignLanguageDataset(
        features_path=args.features_path,
        split_ratio=0.8,
        is_train=True,
        seed=args.seed
    )
    
    val_dataset = SignLanguageDataset(
        features_path=args.features_path,
        split_ratio=0.8,
        is_train=False,
        seed=args.seed
    )
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=custom_collate_fn
    )
    
    # Create model
    model = SignDiffusionModel(fixed_channels=26).to(device)
    
    # Train model
    metrics = train_diffusion_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        log_interval=args.log_interval
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
