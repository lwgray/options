import os
import torch
import argparse
from dataset import AmericanOptionDataset, get_dataloaders
from model_config import get_american_option_config
from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.metrics import relative_lp_error

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
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("Test results:", test_results)
    
    # Save test results
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")
    
    return final_output_dir

if __name__ == "__main__":
    main()