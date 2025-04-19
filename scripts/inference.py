import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import AmericanOptionDataset
from scOT.model import ScOT
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with finetuned Poseidon for American options')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the finetuned model')
    parser.add_argument('--data_dir', type=str, default='poseidon_data',
                        help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='viz/results',
                        help='Directory to save visualization results')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use for visualization')
    
    return parser.parse_args()

def plot_result(input_data, true_output, pred_output, metadata, output_path):
    """Plot comparison between true and predicted option prices"""
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Get grid dimensions
    grid_size = true_output.shape[1]
    X, Y = np.meshgrid(
        np.linspace(0.5, 1.5, grid_size),  # Normalized stock price
        np.linspace(0, 1, grid_size)       # Normalized time
    )
    
    # Plot input parameters - spot/strike ratio
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, input_data[0], cmap='viridis')
    ax1.set_xlabel('Stock Price / Strike')
    ax1.set_ylabel('Time to Expiry')
    ax1.set_zlabel('Value')
    ax1.set_title('Input: Spot/Strike Ratio')
    
    # Plot true option price surface
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, true_output[0], cmap='plasma')
    ax2.set_xlabel('Stock Price / Strike')
    ax2.set_ylabel('Time to Expiry')
    ax2.set_zlabel('Option Price')
    ax2.set_title('True Option Price')
    
    # Plot predicted option price surface
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, pred_output[0], cmap='plasma')
    ax3.set_xlabel('Stock Price / Strike')
    ax3.set_ylabel('Time to Expiry')
    ax3.set_zlabel('Option Price')
    ax3.set_title('Predicted Option Price')
    
    # Add metadata as text
    error = np.mean(np.abs(true_output - pred_output) / (np.abs(true_output) + 1e-6)) * 100
    plt.figtext(0.02, 0.02, 
                f"Strike: {metadata['strike']:.2f}, Spot: {metadata['spot']:.2f}, "
                f"Rate: {metadata['rate']:.2%}, Dividend: {metadata['dividend']:.2%}, "
                f"Volatility: {metadata['volatility']:.2%}, Time: {metadata['time_to_expiry']:.2f} years"
                f"\nMean Relative Error: {error:.2f}%", 
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return error

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = AmericanOptionDataset(args.data_dir, split=args.split)
    
    # Load model
    model = ScOT.from_pretrained(args.model_dir)
    model = model.to(device)
    model.eval()
    
    # Select samples to visualize
    if args.num_samples >= len(dataset):
        sample_indices = list(range(len(dataset)))
    else:
        sample_indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    
    # Run inference and visualize results
    errors = []
    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        input_tensor = sample['pixel_values'].unsqueeze(0) # Add batch dimension
        time_tensor = sample['time'].unsqueeze(0)
        
        # Get metadata
        sample_dir = os.path.join(args.data_dir, dataset.samples[idx])
        with open(os.path.join(sample_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Run inference
        with torch.no_grad():
            output = model(pixel_values=input_tensor, time=time_tensor)
        
        # Convert to numpy for plotting
        input_np = input_tensor.squeeze(0).numpy()
        true_output_np = sample['labels'].numpy()
        pred_output_np = output.output.squeeze(0).numpy()
        
        # Plot and save
        output_path = os.path.join(args.output_dir, f'sample_{idx:04d}.png')
        error = plot_result(input_np, true_output_np, pred_output_np, metadata, output_path)
        errors.append(error)
        
        print(f"Processed sample {idx}, error: {error:.2f}%")
    
    # Print summary statistics
    print(f"\nInference complete for {len(sample_indices)} samples")
    print(f"Mean error: {np.mean(errors):.2f}%")
    print(f"Median error: {np.median(errors):.2f}%")
    print(f"Min error: {np.min(errors):.2f}%")
    print(f"Max error: {np.max(errors):.2f}%")
    
    # Save summary statistics
    with open(os.path.join(args.output_dir, 'inference_results.txt'), 'w') as f:
        f.write(f"Inference results for {len(sample_indices)} samples from {args.split} split\n")
        f.write(f"Mean error: {np.mean(errors):.2f}%\n")
        f.write(f"Median error: {np.median(errors):.2f}%\n")
        f.write(f"Min error: {np.min(errors):.2f}%\n")
        f.write(f"Max error: {np.max(errors):.2f}%\n")

if __name__ == "__main__":
    main()