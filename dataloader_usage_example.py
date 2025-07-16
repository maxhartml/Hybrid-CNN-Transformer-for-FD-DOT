#!/usr/bin/env python3
"""
🚀 NIR DataLoader Usage Example

Quick example of how to use the NIR phantom DataLoader in your training scripts.
Keep this file as a reference for integrating the DataLoader with your ML models.

Author: Max Hart - NIR Tomography Research
"""

import torch
import torch.nn as nn
from code.data_loader import create_nir_dataloaders, TissuePatchCNN

def example_training_setup():
    """Example of how to set up the DataLoader for training."""
    
    # 1. Create dataloaders for all splits
    dataloaders = create_nir_dataloaders(
        data_dir="data",           # Your phantom data directory
        batch_size=32,             # Adjust based on your GPU memory
        num_workers=4,             # Adjust based on your CPU cores
        random_seed=42
    )
    
    # 2. Create CNN encoder for patches
    patch_encoder = TissuePatchCNN(embed_dim=128)
    
    # 3. Access individual dataloaders
    train_loader = dataloaders['train']    # 135,000 samples
    val_loader = dataloaders['val']        # 7,500 samples  
    test_loader = dataloaders['test']      # 7,500 samples
    
    print(f"Training batches per epoch: {len(train_loader):,}")
    
    return dataloaders, patch_encoder

def example_training_loop():
    """Example of a basic training loop with the DataLoader."""
    
    # Setup
    dataloaders, patch_encoder = example_training_setup()
    train_loader = dataloaders['train']
    
    # Your main model (example - replace with your architecture)
    class NIRReconstructionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_encoder = patch_encoder
            self.decoder = nn.Sequential(
                nn.Linear(264, 256),  # 264 = actual token size from DataLoader
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2)     # Output: optical properties
            )
        
        def forward(self, batch):
            # Extract patches and encode them
            source_embedding = self.patch_encoder(batch['source_patch'])
            detector_embedding = self.patch_encoder(batch['detector_patch'])
            
            # Combine into tokens
            tokens = torch.cat([
                batch['geometry'],       # [batch, 6] positions
                batch['measurements'],   # [batch, 2] measurements
                source_embedding,        # [batch, 128] source context
                detector_embedding       # [batch, 128] detector context
            ], dim=1)                   # [batch, 264] complete tokens
            
            # Decode to optical properties
            output = self.decoder(tokens)
            return output
    
    # Create model and optimizer
    model = NIRReconstructionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(3):  # Just 3 epochs for example
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            
            # Forward pass
            predictions = model(batch)
            
            # For this example, we'll use dummy targets
            # In practice, extract target from batch['metadata'] or ground truth
            targets = torch.randn_like(predictions)  # Replace with real targets
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            # Break early for demo
            if batch_idx >= 10:
                break
        
        avg_loss = total_loss / min(11, len(train_loader))
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.6f}")

if __name__ == "__main__":
    print("🚀 NIR DataLoader Usage Example")
    print("=" * 40)
    
    try:
        # Test basic setup
        dataloaders, patch_encoder = example_training_setup()
        print("✅ DataLoader setup successful!")
        
        # Show data statistics
        for split, loader in dataloaders.items():
            print(f"  {split.upper()}: {len(loader.dataset):,} samples")
        
        # Run example training
        print("\nRunning example training loop...")
        example_training_loop()
        print("✅ Example training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
