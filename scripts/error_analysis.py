import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import json
import torch
from dataset import AmericanOptionDataset
from scOT.model import ScOT

def run_error_analysis(model_dir, data_dir, output_dir='error_analysis'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and dataset
    model = ScOT.from_pretrained(model_dir)
    model.eval()
    dataset = AmericanOptionDataset(data_dir, split='test')
    
    # Compute errors for each sample
    errors = []
    metadata_list = []
    surface_errors = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            input_tensor = sample['pixel_values'].unsqueeze(0)
            time_tensor = sample['time'].unsqueeze(0)
            
            # Run inference
            output = model(pixel_values=input_tensor, time=time_tensor)
            
            # Calculate relative error
            pred = output.output.squeeze(0).numpy()
            true = sample['labels'].numpy()
            
            rel_error = np.abs(pred - true) / (np.abs(true) + 1e-6) * 100
            mean_error = np.mean(rel_error)
            errors.append(mean_error)
            surface_errors.append(rel_error)
            
            # Get metadata
            sample_dir = os.path.join(data_dir, dataset.samples[i])
            with open(os.path.join(sample_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                metadata_list.append(metadata)
    
    # Convert to numpy arrays
    errors = np.array(errors)
    surface_errors = np.array(surface_errors)
    
    # 1. Basic stats
    with open(os.path.join(output_dir, 'error_stats.txt'), 'w') as f:
        f.write(f"Error Statistics:\n")
        f.write(f"Mean: {np.mean(errors):.4f}%\n")
        f.write(f"Median: {np.median(errors):.4f}%\n")
        f.write(f"Std Dev: {np.std(errors):.4f}%\n")
        f.write(f"Min: {np.min(errors):.4f}%\n")
        f.write(f"Max: {np.max(errors):.4f}%\n")
        f.write(f"25th Percentile: {np.percentile(errors, 25):.4f}%\n")
        f.write(f"75th Percentile: {np.percentile(errors, 75):.4f}%\n")
    
    # 2. Error distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(errors, bins=20, alpha=0.7, color='blue')
    ax1.axvline(np.median(errors), color='red', linestyle='dashed', linewidth=1, label=f'Median: {np.median(errors):.2f}%')
    ax1.axvline(np.mean(errors), color='green', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(errors):.2f}%')
    ax1.set_xlabel('Relative L1 Error (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution Histogram')
    ax1.legend()
    
    # Density plot
    sns.kdeplot(errors, fill=True, ax=ax2)
    ax2.axvline(np.median(errors), color='red', linestyle='dashed', linewidth=1, label=f'Median: {np.median(errors):.2f}%')
    ax2.axvline(np.mean(errors), color='green', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(errors):.2f}%')
    ax2.set_xlabel('Relative L1 Error (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Error Density Plot')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()
    
    # 3. Parameter correlations
    # Extract parameters
    strikes = np.array([m['strike'] for m in metadata_list])
    spots = np.array([m['spot'] for m in metadata_list])
    moneyness = spots / strikes
    rates = np.array([m['rate'] for m in metadata_list])
    dividends = np.array([m['dividend'] for m in metadata_list])
    volatilities = np.array([m['volatility'] for m in metadata_list])
    times = np.array([m['time_to_expiry'] for m in metadata_list])
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Error': errors,
        'Strike': strikes,
        'Spot': spots,
        'Moneyness': moneyness,
        'Rate': rates,
        'Dividend': dividends,
        'Volatility': volatilities,
        'Time': times
    })
    
    # Compute correlations
    correlations = df.corr()['Error'].drop('Error').sort_values(ascending=False)
    
    # Plot correlations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    params = {
        'Moneyness': moneyness,
        'Strike': strikes, 
        'Rate': rates,
        'Dividend': dividends,
        'Volatility': volatilities,
        'Time': times
    }
    
    for i, (param_name, param_values) in enumerate(params.items()):
        if i < len(axes):
            ax = axes[i]
            ax.scatter(param_values, errors, alpha=0.6)
            
            # Add trend line
            z = np.polyfit(param_values, errors, 1)
            p = np.poly1d(z)
            ax.plot(sorted(param_values), p(sorted(param_values)), "r--")
            
            # Calculate correlation
            correlation = np.corrcoef(param_values, errors)[0, 1]
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Relative L1 Error (%)')
            ax.set_title(f'{param_name} vs Error (r={correlation:.2f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_correlations.png'))
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Parameter Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 4. Analyze high error samples
    top_n = 20
    high_error_indices = np.argsort(errors)[-top_n:]
    high_error_samples = [metadata_list[i] for i in high_error_indices]
    high_errors = errors[high_error_indices]
    
    with open(os.path.join(output_dir, 'high_error_samples.txt'), 'w') as f:
        f.write(f"Top {top_n} highest error samples:\n")
        f.write("-" * 50 + "\n")
        for i, (sample, error) in enumerate(zip(high_error_samples, high_errors)):
            f.write(f"Sample {i+1} - Error: {error:.2f}%\n")
            f.write(f"  Strike: {sample['strike']:.2f}\n")
            f.write(f"  Spot: {sample['spot']:.2f}\n")
            f.write(f"  Moneyness (S/K): {sample['spot']/sample['strike']:.2f}\n")
            f.write(f"  Rate: {sample['rate']:.2%}\n")
            f.write(f"  Dividend: {sample['dividend']:.2%}\n")
            f.write(f"  Volatility: {sample['volatility']:.2%}\n")
            f.write(f"  Time to Expiry: {sample['time_to_expiry']:.2f} years\n")
            f.write("-" * 30 + "\n")
    
    # 5. Group errors by parameter ranges
    # Function to create bins
    def bin_parameter(values, num_bins=4):
        bins = np.linspace(min(values), max(values), num_bins+1)
        bin_indices = np.digitize(values, bins[1:-1])
        return bins, bin_indices
    
    # Create bins for each parameter
    params_to_bin = {
        'Moneyness': moneyness,
        'Volatility': volatilities,
        'Time': times
    }
    
    for param_name, param_values in params_to_bin.items():
        bins, bin_indices = bin_parameter(param_values)
        
        # Calculate average error for each bin
        bin_errors = [errors[bin_indices == i].mean() for i in range(1, len(bins))]
        bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bin_labels, bin_errors)
        
        # Add values on top of bars
        for bar, err in zip(bars, bin_errors):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.1, 
                    f'{err:.2f}%', 
                    ha='center')
        
        plt.xlabel(param_name + ' Range')
        plt.ylabel('Average Error (%)')
        plt.title(f'Error by {param_name} Range')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'error_by_{param_name}.png'))
        plt.close()
    
    # 6. Spatial error analysis for a few samples
    num_viz_samples = 5
    viz_indices = np.random.choice(len(dataset), num_viz_samples, replace=False)
    
    grid_size = surface_errors[0].shape[1]
    X, Y = np.meshgrid(
        np.linspace(0.5, 1.5, grid_size),  # Normalized stock price
        np.linspace(0, 1, grid_size)       # Normalized time
    )
    
    # Also calculate average error surface
    avg_error_surface = np.mean(surface_errors, axis=0)
    
    # Plot average error surface
    fig = plt.figure(figsize=(12, 10))
    
    # 2D heatmap
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(avg_error_surface[0], cmap='hot', origin='lower', aspect='auto',
                   extent=[0.5, 1.5, 0, 1])
    ax1.set_xlabel('Stock Price / Strike')
    ax1.set_ylabel('Time to Expiry')
    ax1.set_title('Average Error Heatmap (%)')
    plt.colorbar(im, ax=ax1)
    
    # 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, avg_error_surface[0], cmap='hot')
    ax2.set_xlabel('Stock Price / Strike')
    ax2.set_ylabel('Time to Expiry')
    ax2.set_zlabel('Average Error (%)')
    ax2.set_title('Average Error Surface')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_error_surface.png'))
    plt.close()
    
    # Individual samples
    for idx in viz_indices:
        sample = dataset[idx]
        input_tensor = sample['pixel_values'].unsqueeze(0)
        time_tensor = sample['time'].unsqueeze(0)
        
        # Get model prediction
        output = model(pixel_values=input_tensor, time=time_tensor)
        
        # Get true and predicted outputs
        true_output = sample['labels'].numpy()
        pred_output = output.output.squeeze(0).numpy()
        
        # Get sample metadata
        sample_dir = os.path.join(data_dir, dataset.samples[idx])
        with open(os.path.join(sample_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Plot
        fig = plt.figure(figsize=(16, 10))
        
        # True price surface
        ax1 = fig.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(X, Y, true_output[0], cmap='viridis')
        ax1.set_xlabel('Stock Price / Strike')
        ax1.set_ylabel('Time to Expiry')
        ax1.set_zlabel('Option Price')
        ax1.set_title('True Option Price')
        
        # Predicted price surface
        ax2 = fig.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(X, Y, pred_output[0], cmap='viridis')
        ax2.set_xlabel('Stock Price / Strike')
        ax2.set_ylabel('Time to Expiry')
        ax2.set_zlabel('Option Price')
        ax2.set_title('Predicted Option Price')
        
        # 2D error heatmap
        ax3 = fig.add_subplot(223)
        error_map = np.abs(true_output - pred_output) / (np.abs(true_output) + 1e-6) * 100
        im3 = ax3.imshow(error_map[0], cmap='hot', origin='lower', aspect='auto', 
                       extent=[0.5, 1.5, 0, 1])
        ax3.set_xlabel('Stock Price / Strike')
        ax3.set_ylabel('Time to Expiry')
        ax3.set_title('Error Heatmap (%)')
        plt.colorbar(im3, ax=ax3)
        
        # 3D error surface
        ax4 = fig.add_subplot(224, projection='3d')
        surf4 = ax4.plot_surface(X, Y, error_map[0], cmap='hot')
        ax4.set_xlabel('Stock Price / Strike')
        ax4.set_ylabel('Time to Expiry')
        ax4.set_zlabel('Relative Error (%)')
        ax4.set_title('Error Surface')
        
        # Add metadata as text
        plt.figtext(0.02, 0.02, 
                    f"Strike: {metadata['strike']:.2f}, Spot: {metadata['spot']:.2f}, "
                    f"Rate: {metadata['rate']:.2%}, Dividend: {metadata['dividend']:.2%}, "
                    f"Volatility: {metadata['volatility']:.2%}, Time: {metadata['time_to_expiry']:.2f} years"
                    f"\nMean Error: {np.mean(error_map):.2f}%, Max Error: {np.max(error_map):.2f}%", 
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Return all the error data
    return {
        'errors': errors,
        'metadata': metadata_list,
        'surface_errors': surface_errors,
        'correlations': correlations
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform error analysis on finetuned model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the finetuned model')
    parser.add_argument('--data_dir', type=str, default='data/poseidon_data',
                        help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='error_analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    results = run_error_analysis(args.model_dir, args.data_dir, args.output_dir)
    print(f"Analysis complete. Results saved to {args.output_dir}")