import os
import torch
import argparse
from dataset import AmericanOptionDataset, get_dataloaders
from model_config import get_american_option_config
from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.metrics import relative_lp_error
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune Poseidon for American options')
    parser.add_argument('--data_dir', type=str, default='poseidon_data',
                        help='Directory containing the formatted Poseidon data')
    parser.add_argument('--output_dir', type=str, default='models/american_options',
                        help='Directory to save model checkpoints')
    parser.add_argument('--model_size', type=str, default='B',
                        choices=['T', 'B', 'L'], help='Poseidon model size')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid size for input/output data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate for main parameters')
    parser.add_argument('--lr_embedding', type=float, default=5e-4,
                        help='Learning rate for embedding/recovery parameters')
    parser.add_argument('--lr_time', type=float, default=5e-4,
                        help='Learning rate for time embedding parameters')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay regularization')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--report_to', type=str, default='none',
                        choices=['none', 'wandb', 'tensorboard', 'all'],
                        help='Where to report training metrics')
    parser.add_argument('--use_all2all', action='store_true',
                        help='Use all2all training strategy')
    
    return parser.parse_args()

def compute_metrics(eval_preds):
    """Compute evaluation metrics for American option model"""
    import numpy as np
    
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    
    # Calculate relative L1 error (as percentage)
    errors = relative_lp_error(
        predictions,
        labels,
        p=1,
        return_percent=True
    )
    
    # Calculate statistics across batch
    median_error = np.median(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    
    # Return metrics dictionary
    return {
        "median_relative_l1_error": median_error,
        "mean_relative_l1_error": mean_error,
        "std_relative_l1_error": std_error,
        "max_relative_l1_error": max_error,
    }

def train_with_all2all_strategy(model, train_loader, val_loader, optimizer, device, epochs, output_dir):
    """
    Train the model using the all2all strategy where we create multiple time pairs
    from each trajectory for more efficient learning.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        device: Device to train on
        epochs: Number of training epochs
        output_dir: Directory to save checkpoints
        
    Returns:
        Trained model
    """
    import time
    from tqdm import tqdm
    import numpy as np
    
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            # Extract data
            input_tensor = batch['pixel_values'].to(device)  # Shape: [B, C, H, W]
            output_tensor = batch['labels'].to(device)       # Shape: [B, C, H, W]
            
            # For all2all strategy, we consider all combinations of timesteps
            # In this simplified version, we're using the initial and final state
            # and creating intermediate states through interpolation
            
            # Generate multiple timesteps (e.g., 5 steps)
            num_steps = 5
            timesteps = torch.linspace(0.0, 1.0, num_steps).to(device)
            
            # Create interpolated states for each timestep
            states = []
            for t in timesteps:
                # Linear interpolation between input and output
                state = (1-t) * input_tensor + t * output_tensor
                states.append(state)
            
            # Process all combinations of timesteps (i, j) where i < j
            batch_loss = 0.0
            pair_count = 0
            
            for i in range(num_steps-1):
                for j in range(i+1, num_steps):
                    # Get states at time i and j
                    state_i = states[i]
                    state_j = states[j]
                    
                    # Calculate time difference for model
                    time_diff = timesteps[j] - timesteps[i]
                    
                    # Forward pass: predict from time i to time j
                    outputs = model(
                        pixel_values=state_i,
                        time=time_diff.expand(input_tensor.size(0), 1)
                    )
                    
                    # Calculate loss (relative L1)
                    epsilon = 1e-10  # Prevent division by zero
                    pred = outputs.output
                    diff = torch.abs(pred - state_j)
                    denom = torch.abs(state_j) + epsilon
                    loss = torch.mean(diff / denom)
                    
                    batch_loss += loss
                    pair_count += 1
            
            # Average loss over all pairs
            if pair_count > 0:
                avg_loss = batch_loss / pair_count
                
                # Backward pass
                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()
                
                train_loss += avg_loss.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": avg_loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        train_time = time.time() - start_time
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        start_time = time.time()
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in progress_bar:
                # Standard validation - direct evaluation
                input_tensor = batch['pixel_values'].to(device)
                output_tensor = batch['labels'].to(device)
                time_tensor = batch['time'].to(device)
                
                # Forward pass
                outputs = model(pixel_values=input_tensor, time=time_tensor)
                
                # Calculate relative L1 loss
                epsilon = 1e-10
                pred = outputs.output
                diff = torch.abs(pred - output_tensor)
                denom = torch.abs(output_tensor) + epsilon
                loss = torch.mean(diff / denom)
                
                val_loss += loss.item()
                val_batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        val_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f} ({train_time:.2f}s), "
              f"Val Loss: {avg_val_loss:.6f} ({val_time:.2f}s)")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save model checkpoint
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            best_model_path = checkpoint_dir
            print(f"New best model saved to {checkpoint_dir}")
    
    # Load best model
    if best_model_path:
        print(f"Loading best model from {best_model_path}")
        model = ScOT.from_pretrained(best_model_path)
        model = model.to(device)
    
    return model

def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading datasets from {args.data_dir}")
    # Load datasets
    train_dataset = AmericanOptionDataset(args.data_dir, split='train')
    val_dataset = AmericanOptionDataset(args.data_dir, split='val')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model configuration
    model_config = get_american_option_config(grid_size=args.grid_size, model_size=args.model_size)
    
    # Set device
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    device = torch.device("mps" if mps_available else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    pretrained_model_name = f"camlab-ethz/Poseidon-{args.model_size}"
    print(f"Loading pretrained model: {pretrained_model_name}")
    model = ScOT.from_pretrained(
        pretrained_model_name,
        config=model_config,
        ignore_mismatched_sizes=True
    )

    # Move model to device
    model = model.to(device)
    
    print(f"Model loaded: {model_config.num_channels} input channels, {model_config.num_out_channels} output channels")
    
    if args.use_all2all:
        print("Using all2all training strategy")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if not ('embeddings' in n or 'patch_recovery' in n or '.norm' in n)], 'lr': args.lr},
            {'params': [p for n, p in model.named_parameters() if 'embeddings' in n or 'patch_recovery' in n], 'lr': args.lr_embedding},
            {'params': [p for n, p in model.named_parameters() if '.norm' in n], 'lr': args.lr_time}
        ], weight_decay=args.weight_decay)
        
        # Train with all2all strategy
        model = train_with_all2all_strategy(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            output_dir=args.output_dir
        )
        
        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        model.save_pretrained(final_output_dir)
        print(f"Final model saved to {final_output_dir}")
        
    else:
        print("Using standard training")
        # Configure training
        training_args = TrainingArguments(
            use_mps_device=mps_available,
            output_dir=args.output_dir,
            learning_rate=args.lr,
            learning_rate_embedding_recovery=args.lr_embedding,
            learning_rate_time_embedding=args.lr_time,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="mean_relative_l1_error",
            greater_is_better=False,
            save_total_limit=3,
            dataloader_num_workers=args.num_workers,
            report_to=args.report_to,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Start training
        print(f"Starting training for {args.epochs} epochs")
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        trainer.save_model(final_output_dir)
        print(f"Model saved to {final_output_dir}")
    
    # Evaluate on test dataset
    test_dataset = AmericanOptionDataset(args.data_dir, split='test')
    
    # Use standard trainer for evaluation regardless of training method
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
    )
    
    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )
    
    test_results = eval_trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("Test results:", test_results)
    
    # Save test results
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")
    
    return final_output_dir

if __name__ == "__main__":
    main()