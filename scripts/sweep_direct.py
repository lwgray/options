#!/usr/bin/env python
# sweep_direct.py - Run multiple Poseidon finetuning configurations in the same process

import os
import argparse
import time
import torch
from datetime import datetime
import gc

from model_config import get_american_option_config
from dataset import AmericanOptionDataset
from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.metrics import relative_lp_error

def parse_args():
    parser = argparse.ArgumentParser(description='Run multiple finetuning configurations')
    parser.add_argument('--data_dir', type=str, default='data/poseidon_data',
                        help='Directory containing the formatted Poseidon data')
    parser.add_argument('--output_dir', type=str, default='models/american_options_sweep',
                        help='Base directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for sweep configurations')
    parser.add_argument('--report_to', type=str, default='none',
                        choices=['none', 'wandb', 'tensorboard', 'all'],
                        help='Where to report training metrics')
    parser.add_argument('--max_samples_pct', type=float, default=1.0,
                        help='Maximum percentage of samples to load')
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
    
    # Calculate statistics
    median_error = np.median(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    
    return {
        "median_relative_l1_error": median_error,
        "mean_relative_l1_error": mean_error,
        "std_relative_l1_error": std_error,
        "max_relative_l1_error": max_error,
    }

def run_training(config, args, run_dir, logfile, run_index, total_runs):
    """Run a single training configuration"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    
    # Log start of run
    with open(logfile, 'a') as f:
        f.write(f"\n\n===== Running configuration {run_index}/{total_runs} =====\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n\n===== Running configuration {run_index}/{total_runs} =====")
    print(f"Configuration: {config}")
    print(f"Output directory: {run_dir}")
    start_time = time.time()
    
    # Initialize wandb if using it
    if args.report_to == "wandb" or args.report_to == "all":
        import wandb
        
        # Make sure any previous run is finished
        if wandb.run is not None:
            wandb.finish()
        
        # Start a new run with a unique name
        run_name = f"run_{run_index}_lr{config['lr']}_emb{config['lr_embedding']}_batch{config['batch_size']}"
        wandb.init(project="american_options", name=run_name, config={
            "lr": config["lr"],
            "lr_embedding": config["lr_embedding"],
            "lr_time": config["lr_time"],
            "batch_size": config["batch_size"],
            "warmup_ratio": config.get("warmup_ratio", 0.0),
            "model_size": "T",  # Changed from "B" to match the model being loaded
            "grid_size": 64,
            "epochs": 25,
            "weight_decay": 1e-6,
            "run_index": run_index,
            "total_runs": total_runs,
        })
    
    try:
        # Load datasets - you only need to do this once if the datasets are the same for all runs
        print(f"Loading datasets from {args.data_dir}")
        train_dataset = AmericanOptionDataset(args.data_dir, split='train')
        val_dataset = AmericanOptionDataset(args.data_dir, split='val')
        test_dataset = AmericanOptionDataset(args.data_dir, split='test')
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create model configuration
        model_config = get_american_option_config(grid_size=64, model_size="T")
        
        # Load pretrained model
        pretrained_model_name = "camlab-ethz/Poseidon-T"
        print(f"Loading pretrained model: {pretrained_model_name}")
        model = ScOT.from_pretrained(
            pretrained_model_name,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        
        # Configure training
        training_args = TrainingArguments(
            output_dir=run_dir,
            learning_rate=config["lr"],
            learning_rate_embedding_recovery=config["lr_embedding"],
            learning_rate_time_embedding=config["lr_time"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=25,
            weight_decay=1e-6,
            load_best_model_at_end=True,
            metric_for_best_model="median_relative_l1_error",
            greater_is_better=False,
            save_total_limit=3,
            dataloader_num_workers=1,
            report_to=args.report_to,
            warmup_ratio=config.get("warmup_ratio", 0.0),
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
        print(f"Starting training for 25 epochs")
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(run_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        trainer.save_model(final_output_dir)
        print(f"Model saved to {final_output_dir}")
        
        # Evaluate on test dataset
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        print("Test results:", test_results)
        
        # Save test results
        with open(os.path.join(run_dir, "test_results.txt"), "w") as f:
            for key, value in test_results.items():
                f.write(f"{key}: {value}\n")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Log success
        with open(logfile, 'a') as f:
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write("Status: Success\n")
            f.write("Test results:\n")
            for key, value in test_results.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Run completed successfully in {duration:.2f} seconds")
        print(f"Test results: {test_results}")
        
        # Finish the wandb run if using it
        if args.report_to == "wandb" or args.report_to == "all":
            import wandb
            if wandb.run is not None:
                wandb.finish()
        
        return True
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        # Log failure
        with open(logfile, 'a') as f:
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write("Status: Failed\n")
            f.write(f"Error: {str(e)}\n")
        
        print(f"Run failed with error: {e}")
        
        # Make sure to finish the wandb run even if there's an error
        if args.report_to == "wandb" or args.report_to == "all":
            import wandb
            if wandb.run is not None:
                wandb.finish()
        
        return False

def main():
    args = parse_args()
    
    # Define sweep configurations
    sweep_configs = [
        # Learning rate variations
        {"lr": 1e-5, "lr_embedding": 5e-4, "lr_time": 5e-4, "batch_size": 16, "warmup_ratio": 0.0},
        {"lr": 2e-5, "lr_embedding": 5e-4, "lr_time": 5e-4, "batch_size": 16, "warmup_ratio": 0.0},
        {"lr": 5e-5, "lr_embedding": 5e-4, "lr_time": 5e-4, "batch_size": 16, "warmup_ratio": 0.0},
        
        # Embedding learning rate variations
        {"lr": 5e-5, "lr_embedding": 1e-3, "lr_time": 5e-4, "batch_size": 16, "warmup_ratio": 0.0},
        {"lr": 5e-5, "lr_embedding": 2e-3, "lr_time": 5e-4, "batch_size": 16, "warmup_ratio": 0.0},
        
        # Time embedding learning rate variations
        {"lr": 5e-5, "lr_embedding": 5e-4, "lr_time": 1e-3, "batch_size": 16, "warmup_ratio": 0.0},
        
        # Warmup ratio variations
        {"lr": 5e-5, "lr_embedding": 5e-4, "lr_time": 5e-4, "batch_size": 16, "warmup_ratio": 0.05},
        {"lr": 5e-5, "lr_embedding": 5e-4, "lr_time": 5e-4, "batch_size": 16, "warmup_ratio": 0.1},
        
        # Batch size variations
        {"lr": 5e-5, "lr_embedding": 5e-4, "lr_time": 5e-4, "batch_size": 32, "warmup_ratio": 0.0},
    ]
    
    # Create base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a log file to track progress
    logfile = os.path.join(args.output_dir, f"sweep_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Run each configuration
    for i, config in enumerate(sweep_configs[args.start_idx:], args.start_idx):
        # Create a run-specific output directory
        run_dir = os.path.join(args.output_dir, f"run_{i+1}_lr{config['lr']}_emb{config['lr_embedding']}_batch{config['batch_size']}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Run the configuration
        success = run_training(config, args, run_dir, logfile, i+1, len(sweep_configs))
        
        # Force garbage collection to free memory
        gc.collect()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Wait a bit between runs to ensure resources are freed
        if i < len(sweep_configs) - 1:
            print("Waiting 5 seconds before next run...")
            time.sleep(5)
    
    print(f"\nAll configurations completed. See log file at {logfile}")

if __name__ == "__main__":
    main()