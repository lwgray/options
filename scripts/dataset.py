import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader


class AmericanOptionDataset(Dataset):
    """Dataset for American option pricing"""

    def __init__(self, data_dir, split='train', all2all=True):
        """
        Args:
            data_dir (str): Directory with all the data
            split (str): Split to load ('train', 'val', 'test')
            all2all (bool): Whether to return all time steps for all2all training
        """
        self.data_dir = data_dir
        self.all2all = all2all

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
            dict: When all2all=True, returns all time steps for the sample
                  When all2all=False, returns just the initial and final states
        """
        sample_dir = os.path.join(self.data_dir, self.samples[idx])

        if self.all2all:
            # Load all available time steps for this sample
            #This implementation assumes you have timesteps stored in separate files
            # Adapt this to your storage format

            time_steps = []
            inputs = []

            # Assume time steps are stored in files named t_0.npy, t_1.npy, etc.
            time_files = sorted(glob.glob(os.path.join(sample_dir, 't_*.npy')))

            for time_file in time_files:
                time_step = float(time_file.split('_')[-1].split('.')[0])
                data = np.load(os.path.join(sample_dir, time_file))

                time_steps.append(time_step)
                inputs.append(torch.from_numpy(data).float())
            
            input_tensor = torch.stack(inputs)
            time_tensor = torch.tensor(time_steps, dtype=torch.float)

            return {
                'pixel_values': input_tensor,
                'time': time_tensor
            }
        else:
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