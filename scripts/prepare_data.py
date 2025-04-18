import numpy as np
import os
import json
import matplotlib.pyplot as plt
from generate_data import load_dataset

def prepare_for_poseidon(inputs, solutions, output_dir='data/poseidon_data'):
    """
    Format the American option data for use with Poseidon
    
    Args:
        inputs: Dictionary of input parameters
        solutions: Array of price surfaces
        output_dir: Directory to save the formatted data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_samples = len(inputs['strike'])
    grid_size = solutions.shape[1]
    
    # For each sample, we need to create:
    # 1. A set of input parameters formatted as a function on a grid
    # 2. The corresponding price surface as the output function
    
    for i in range(num_samples):
        # Create a more informative directory structure
        sample_dir = os.path.join(output_dir, f"sample_{i:04d}")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        # For Poseidon, we need to format inputs as a grid
        # We'll create a 4-channel input: [spot/strike, rate, dividend, volatility]
        input_grid = np.zeros((4, grid_size, grid_size))
        
        # Normalized spot prices (x-axis of our grid)
        spot_points = np.linspace(0.5, 1.5, grid_size)
        
        # Normalized time points (y-axis of our grid)
        time_points = np.linspace(0, 1, grid_size)
        
        # Create parameter grids
        for x in range(grid_size):
            for y in range(grid_size):
                # Channel 1: spot/strike ratio at this grid point
                spot = inputs['spot'][i] * spot_points[x]
                input_grid[0, y, x] = spot / inputs['strike'][i]
                
                # Channel 2: risk-free rate
                input_grid[1, y, x] = inputs['rate'][i]
                
                # Channel 3: dividend yield
                input_grid[2, y, x] = inputs['dividend'][i]
                
                # Channel 4: volatility
                input_grid[3, y, x] = inputs['volatility'][i]
        
        # Save the input and output grids
        np.save(os.path.join(sample_dir, 'input.npy'), input_grid)
        # For output, we need to reshape to have channel dimension first (1, H, W)
        output_grid = solutions[i].reshape(1, grid_size, grid_size)
        np.save(os.path.join(sample_dir, 'output.npy'), output_grid)
        
        # Save metadata
        metadata = {
            'strike': float(inputs['strike'][i]),
            'spot': float(inputs['spot'][i]),
            'rate': float(inputs['rate'][i]),
            'dividend': float(inputs['dividend'][i]),
            'volatility': float(inputs['volatility'][i]),
            'time_to_expiry': float(inputs['time_to_expiry'][i]),
        }
        
        with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Prepared {num_samples} samples for Poseidon in {output_dir}")
    
    # Create a dataset split file
    splits = {
        'train': [f"sample_{i:04d}" for i in range(int(num_samples * 0.7))],
        'val': [f"sample_{i:04d}" for i in range(int(num_samples * 0.7), int(num_samples * 0.85))],
        'test': [f"sample_{i:04d}" for i in range(int(num_samples * 0.85), num_samples)]
    }
    
    with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    # Create a visualization of the dataset
    visualize_dataset(inputs, solutions, output_dir)

def visualize_dataset(inputs, solutions, output_dir):
    """Create visualization of dataset statistics"""
    os.makedirs(os.path.join(output_dir, 'viz'), exist_ok=True)
    
    # Plot histograms of parameters
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].hist(inputs['strike'], bins=20)
    axs[0, 0].set_title('Strike Price Distribution')
    
    axs[0, 1].hist(inputs['spot'], bins=20)
    axs[0, 1].set_title('Spot Price Distribution')
    
    axs[0, 2].hist(inputs['rate'] * 100, bins=20)
    axs[0, 2].set_title('Risk-Free Rate Distribution (%)')
    
    axs[1, 0].hist(inputs['dividend'] * 100, bins=20)
    axs[1, 0].set_title('Dividend Yield Distribution (%)')
    
    axs[1, 1].hist(inputs['volatility'] * 100, bins=20)
    axs[1, 1].set_title('Volatility Distribution (%)')
    
    axs[1, 2].hist(inputs['time_to_expiry'] * 365, bins=20)
    axs[1, 2].set_title('Time to Expiry (days)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'viz', 'parameter_distributions.png'))
    plt.close()
    
    # Plot a few random samples
    indices = np.random.choice(len(inputs['strike']), 3, replace=False)
    
    for i, idx in enumerate(indices):
        fig = plt.figure(figsize=(12, 6))
        
        # 3D surface
        ax1 = fig.add_subplot(121, projection='3d')
        grid_size = solutions.shape[1]
        X, Y = np.meshgrid(
            np.linspace(0.5, 1.5, grid_size),  # Normalized stock price
            np.linspace(0, 1, grid_size)       # Normalized time
        )
        surf = ax1.plot_surface(X, Y, solutions[idx], cmap='viridis')
        ax1.set_xlabel('Stock Price / Spot')
        ax1.set_ylabel('Time to Expiry')
        ax1.set_zlabel('Option Price')
        ax1.set_title('American Option Price Surface')
        
        # Parameters
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        parameter_text = f"""
        Strike: {inputs['strike'][idx]:.2f}
        Spot: {inputs['spot'][idx]:.2f}
        Risk-Free Rate: {inputs['rate'][idx]:.2%}
        Dividend Yield: {inputs['dividend'][idx]:.2%}
        Volatility: {inputs['volatility'][idx]:.2%}
        Time to Expiry: {inputs['time_to_expiry'][idx]:.2f} years
        """
        ax2.text(0.1, 0.5, parameter_text, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'viz', f'sample_{idx:04d}.png'))
        plt.close()
    
    print(f"Visualizations saved to {os.path.join(output_dir, 'viz')}")

if __name__ == "__main__":
    print("Preparing American option dataset for Poseidon...")
    
    # Load the dataset
    inputs, solutions = load_dataset()
    
    # Prepare for Poseidon
    prepare_for_poseidon(inputs, solutions)