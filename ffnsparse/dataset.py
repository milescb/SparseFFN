import torch
from torch.utils.data import Dataset, DataLoader

import random

class RegressionDataset(Dataset):
    """Dataset class for input-target pairs"""
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        input_data, target = self.data_pairs[idx]
        return input_data, target

def training_data(filename, train_test_split=0.8, random_seed=None, return_datasets=False):
    """
    Load data and split into training and testing sets
    
    Args:
        filename: Path to the saved dataset
        train_test_split: Ratio for train/test split
        random_seed: Random seed for reproducibility
        return_datasets: If True, returns Dataset objects instead of lists
    
    Returns:
        train_data, test_data: Either lists or Dataset objects depending on return_datasets
    """
    data = torch.load(filename)
    
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create a randomly shuffled copy of the data
    shuffled_indices = list(range(len(data)))
    random.shuffle(shuffled_indices)
    shuffled_data = [data[i] for i in shuffled_indices]
    
    # split data into training and testing
    split = int(len(shuffled_data) * train_test_split)
    train_data = shuffled_data[:split]
    test_data = shuffled_data[split:]
    
    if return_datasets:
        train_dataset = RegressionDataset(train_data)
        test_dataset = RegressionDataset(test_data)
        return train_dataset, test_dataset, split
    
    return train_data, test_data, split

def get_data_loaders(filename, batch_size=32, train_test_split=0.8, random_seed=None):
    """
    Load data and create DataLoaders for training and testing
    
    Args:
        filename: Path to the saved dataset
        batch_size: Batch size for DataLoader
        train_test_split: Ratio for train/test split
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    # Get datasets
    train_dataset, test_dataset, split_indices = training_data(
        filename, 
        train_test_split=train_test_split,
        random_seed=random_seed,
        return_datasets=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader, split_indices

def testing_data(filename, indices):
    """Load specific data samples by indices"""
    data = torch.load(filename)
    selected_data_input = [data[i][0] for i in indices]
    selected_data_target = [data[i][1] for i in indices]
    return selected_data_input, selected_data_target