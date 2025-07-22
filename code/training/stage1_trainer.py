"""
Simple Stage 1 Trainer: CNN Autoencoder Pre-training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from ..models.hybrid_model import HybridCNNTransformer
from .training_utils import RMSELoss, compute_rmse


class Stage1Trainer:
    """Simple Stage 1 trainer for CNN autoencoder pre-training"""
    
    def __init__(self, learning_rate=1e-4, device="cpu"):
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        
        # Initialize model (no tissue patches for stage 1)
        self.model = HybridCNNTransformer(use_tissue_patches=False)
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"Stage 1 Trainer initialized on {self.device}")
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            measurements = batch['measurements'].to(self.device)
            targets = batch['volumes'].to(self.device)
            
            # Forward pass (no tissue patches for stage 1)
            self.optimizer.zero_grad()
            outputs = self.model(measurements, tissue_patches=None)
            
            # Compute loss
            loss = self.criterion(outputs['reconstructed'], targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, data_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                measurements = batch['measurements'].to(self.device)
                targets = batch['volumes'].to(self.device)
                
                outputs = self.model(measurements, tissue_patches=None)
                loss = self.criterion(outputs['reconstructed'], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, data_loaders, epochs=50):
        """Train the model"""
        print(f"🏋️ Starting Stage 1 training for {epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(data_loaders['train'])
            
            # Validate  
            val_loss = self.validate(data_loaders['val'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"checkpoints/stage1_best.pth", epoch, val_loss)
        
        print(f"✅ Stage 1 training complete! Best val loss: {best_val_loss:.6f}")
    
    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
        print(f"💾 Saved checkpoint: {path}")
