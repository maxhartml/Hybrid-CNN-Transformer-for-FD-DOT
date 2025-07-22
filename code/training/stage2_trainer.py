"""
Simple Stage 2 Trainer: Transformer training with frozen CNN decoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from ..models.hybrid_model import HybridCNNTransformer
from .training_utils import RMSELoss, compute_rmse


class Stage2Trainer:
    """Simple Stage 2 trainer for transformer with frozen CNN decoder"""
    
    def __init__(self, stage1_checkpoint_path, use_tissue_patches=True, learning_rate=5e-5, device="cpu"):
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.use_tissue_patches = use_tissue_patches
        
        # Initialize model
        self.model = HybridCNNTransformer(use_tissue_patches=use_tissue_patches)
        self.model.to(self.device)
        
        # Load Stage 1 checkpoint
        self.load_stage1_checkpoint(stage1_checkpoint_path)
        
        # Freeze CNN decoder (Robin Dale's approach)
        self.freeze_cnn_decoder()
        
        # Loss and optimizer (only for unfrozen parameters)
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        
        print(f"Stage 2 Trainer initialized on {self.device}")
        print(f"Use tissue patches: {use_tissue_patches}")
    
    def load_stage1_checkpoint(self, checkpoint_path):
        """Load Stage 1 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 Loaded Stage 1 checkpoint: {checkpoint_path}")
    
    def freeze_cnn_decoder(self):
        """Freeze CNN decoder parameters following Robin Dale's approach"""
        # Freeze the entire CNN autoencoder
        for param in self.model.cnn_autoencoder.parameters():
            param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"🔒 CNN decoder frozen. Trainable params: {trainable_params}/{total_params}")
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            measurements = batch['measurements'].to(self.device)
            targets = batch['volumes'].to(self.device)
            
            # Get tissue patches if using them
            tissue_patches = None
            if self.use_tissue_patches:
                tissue_patches = batch['tissue_patches'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(measurements, tissue_patches)
            
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
                
                tissue_patches = None
                if self.use_tissue_patches:
                    tissue_patches = batch['tissue_patches'].to(self.device)
                
                outputs = self.model(measurements, tissue_patches)
                loss = self.criterion(outputs['reconstructed'], targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, data_loaders, epochs=100):
        """Train the model"""
        mode = "Enhanced" if self.use_tissue_patches else "Baseline"
        print(f"🏋️ Starting Stage 2 training ({mode}) for {epochs} epochs")
        
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
                mode_suffix = "enhanced" if self.use_tissue_patches else "baseline"
                self.save_checkpoint(f"checkpoints/stage2_{mode_suffix}_best.pth", epoch, val_loss)
        
        print(f"✅ Stage 2 training complete! Best val loss: {best_val_loss:.6f}")
    
    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'use_tissue_patches': self.use_tissue_patches
        }, path)
        print(f"💾 Saved checkpoint: {path}")
