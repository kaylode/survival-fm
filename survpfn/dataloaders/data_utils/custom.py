import torch
from torch.utils.data import Dataset

class TorchSurvivalDataset(Dataset):
    def __init__(self, x, t, e):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.e = torch.tensor(e, dtype=torch.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.e[idx], idx  # idx for tracking

class TorchSurvivalDatasetDeepHit(Dataset):
    """Dataset class for DeepHit model"""
    def __init__(self, x, t, e, num_Category):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32).view(-1, 1)
        self.e = torch.tensor(e, dtype=torch.float32).view(-1, 1)
        
        # Create discretized time if needed
        self.t_discrete = torch.floor(self.t * num_Category / torch.max(self.t)).clamp(0, num_Category-1).long()
        
        # Create masks for loss calculation
        self.num_Category = num_Category
        # ensure e is numeric to find max
        self.num_Event = int(torch.max(self.e).item())
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.e[idx], self.t_discrete[idx]
