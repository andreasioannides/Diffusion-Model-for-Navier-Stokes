from models.network import UNet
import torch

SEED_NUM = 42
torch.manual_seed(SEED_NUM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


model = UNet().to(device)
image = torch.full((1, 4, 50, 50), 10.)
model = UNet()
time_emb = torch.tensor([[[5.]]])
print(model(image, time_emb))
