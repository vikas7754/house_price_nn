import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class HousePriceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # Select features and target
        self.features = self.df[[
            'Avg. Area Income',
            'Avg. Area House Age',
            'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms',
            'Area Population'
        ]].values.astype('float32')
        
        self.targets = self.df['Price'].values.astype('float32').reshape(-1, 1)
        
        # Normalize features
        self.feature_mean = torch.tensor(self.features.mean(axis=0))
        self.feature_std = torch.tensor(self.features.std(axis=0))
        self.features = (self.features - self.feature_mean.numpy()) / (self.feature_std.numpy() + 1e-6)
        
        # Normalize targets
        self.target_mean = torch.tensor(self.targets.mean(axis=0))
        self.target_std = torch.tensor(self.targets.std(axis=0))
        self.targets = (self.targets - self.target_mean.numpy()) / (self.target_std.numpy() + 1e-6)
        
        # Save stats
        torch.save({
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }, 'training_stats.pt')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

def get_dataloader(csv_file='housing.csv', batch_size=32, pin_memory=False, **kwargs):
    dataset = HousePriceDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, **kwargs)
