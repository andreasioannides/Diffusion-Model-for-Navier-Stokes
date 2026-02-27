from pathlib import Path
import os
import yaml
from models.network import UNet
from training.train import DiffusionModel
import torch
from torch.utils.data import Dataset, DataLoader


SEED_NUM = 42
torch.manual_seed(SEED_NUM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

PROJECT_ROOT = Path(__file__).resolve().parent
config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config_file = yaml.safe_load(file)


class PDEsDataset(Dataset):
    def __init__(self):
        data = torch.full((64, 4, 50, 50), 10.)
        self.data = torch.tensor(data)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]  
    

batch_size = config_file['training']['batch_size']
n_epochs = config_file['training']['epochs']
PROJECT_ROOT = Path(__file__).resolve().parent.parent
model_path = os.path.join(PROJECT_ROOT, 'models', 'model.pth')
ema_model_path = os.path.join(PROJECT_ROOT, 'models', 'ema_model.pth')

trainset = PDEsDataset()
train_dataloader = DataLoader(trainset, batch_size=batch_size)
diffusion_model = DiffusionModel()
model = UNet().to(device)
# time_emb = torch.tensor([[[5.]]])

diffusion_model.train(model, train_dataloader, n_epochs, model_path, ema_model_path)
