#!/usr/bin/env python3
"""
Simple training script for Robin Dale's hybrid CNN-Transformer approach.
Two-stage training: CNN pre-training → Transformer with frozen decoder
"""

import torch
import argparse
from pathlib import Path

# Import our components
from code.data_processing.data_loader import create_nir_dataloaders
from code.training.stage1_trainer import Stage1Trainer
from code.training.stage2_trainer import Stage2Trainer
from code.training.training_utils import RMSELoss

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['stage1', 'stage2'], required=True,
                       help='Training stage to run')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--use_tissue_patches', action='store_true',
                       help='Use tissue patches (only for stage2)')
    parser.add_argument('--stage1_checkpoint', type=str,
                       help='Path to stage1 checkpoint (required for stage2)')
    args = parser.parse_args()

    print(f"🔬 Robin Dale's Hybrid Training Pipeline")
    print(f"Stage: {args.stage}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    
    if args.stage == 'stage2':
        print(f"Use tissue patches: {args.use_tissue_patches}")

    # Load data
    print("📊 Loading data...")
    data_loaders = create_nir_dataloaders(
        data_dir="data",
        batch_size=BATCH_SIZE,
        use_tissue_patches=args.use_tissue_patches if args.stage == 'stage2' else False
    )

    # Train based on stage
    if args.stage == 'stage1':
        print("🏋️ Starting Stage 1: CNN Autoencoder Pre-training")
        trainer = Stage1Trainer(
            learning_rate=LEARNING_RATE,
            device=DEVICE
        )
        trainer.train(data_loaders, epochs=args.epochs)
        
    elif args.stage == 'stage2':
        if not args.stage1_checkpoint:
            raise ValueError("Stage 2 requires --stage1_checkpoint path")
            
        print("🏋️ Starting Stage 2: Transformer Training")
        print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        
        trainer = Stage2Trainer(
            stage1_checkpoint_path=args.stage1_checkpoint,
            use_tissue_patches=args.use_tissue_patches,
            learning_rate=LEARNING_RATE,
            device=DEVICE
        )
        trainer.train(data_loaders, epochs=args.epochs)

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
