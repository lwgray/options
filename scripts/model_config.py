from scOT.model import ScOTConfig

def get_american_option_config(grid_size=64, model_size='B'):
    """
    Create a ScOTConfig for American option pricing
    
    Args:
        grid_size: Size of the grid for input/output data
        model_size: Size of the Poseidon model ('T', 'B', or 'L')
    
    Returns:
        ScOTConfig object configured for American option data
    """
    # Set model parameters based on size
    if model_size == 'T':
        embed_dim = 48
        depths = [4, 4, 4, 4]
    elif model_size == 'B':
        embed_dim = 96
        depths = [8, 8, 8, 8]
    elif model_size == 'L':
        embed_dim = 192
        depths = [8, 8, 8, 8]
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose 'T', 'B', or 'L'")
    
    # Create configuration
    config = ScOTConfig(
        # Basic image/grid configuration
        image_size=grid_size,      # Size of your grid (e.g., 64x64)
        patch_size=4,              # Size of patches for vision transformer
        
        # Channel configuration
        num_channels=4,            # 4 input channels: [spot/strike, rate, dividend, volatility]
        num_out_channels=1,        # 1 output channel: option price
        
        # Model architecture
        embed_dim=embed_dim,       # Embedding dimension
        depths=depths,             # Number of transformer blocks at each level
        num_heads=[3, 6, 12, 24],  # Number of attention heads at each level
        skip_connections=[2, 2, 2, 0],  # Skip connections between encoder/decoder
        window_size=16,            # Window size for shifted window attention
        
        # Model behavior
        use_conditioning=True,     # Enable time conditioning (important for PDEs)
        
        # Channel normalization for loss computation
        channel_slice_list_normalized_loss=[0, 1],  # Boundaries for loss calculation
        
        # Other parameters
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        residual_model="convnext",
        
        # Loss function
        p=1,  # L1 loss for financial data (more robust to outliers)
    )
    
    return config

if __name__ == "__main__":
    # Example usage
    config = get_american_option_config(grid_size=64, model_size='B')
    print(f"Created configuration with {config.num_channels} input channels and {config.num_out_channels} output channels")
    print(f"Embedding dimension: {config.embed_dim}")
    print(f"Model depths: {config.depths}")