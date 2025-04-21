import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader


class AmericanOptionDataset(Dataset):
    """Dataset for American option pricing"""

    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir (str): Directory with all the data
            split (str): Split to load ('train', 'val', 'test')
            all2all (bool): Whether to return all time steps for all2all training
        """
        self.data_dir = data_dir

        # Load split information
        with open(os.path.join(data_dir, 'splits.json'), 'r') as f:
            splits = json.load(f)

        self.samples = splits[split]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            dict: (pixel_values, labels, time)
        """
        sample_dir = os.path.join(self.data_dir, self.samples[idx])

        # Load input and output
    
        input_data = np.load(os.path.join(sample_dir, 'input.npy'))
        output_data = np.load(os.path.join(sample_dir, 'output.npy'))

        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).float().cpu()
        output_tensor = torch.from_numpy(output_data).float().cpu()
        time = 1.0

        return {'pixel_values': input_tensor,
                'labels': output_tensor,
                'time': torch.tensor([time], dtype=torch.float).cpu()
                }
    
def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """Create dataloaders for training, validation, and testing"""
    
    # Create datasets
    train_dataset = AmericanOptionDataset(data_dir, split='train')
    val_dataset = AmericanOptionDataset(data_dir, split='val')
    test_dataset = AmericanOptionDataset(data_dir, split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test dataset loading
    data_dir = "data/poseidon_data"
    dataset = AmericanOptionDataset(data_dir, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Input shape: {sample['pixel_values'].shape}")
    print(f"Output shape: {sample['labels'].shape}")
    print(f"Time value: {sample['time']}")
    
    # Test dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=16)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")