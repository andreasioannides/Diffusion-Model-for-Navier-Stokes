from pathlib import Path
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from typing import Union, Tuple, List
import math


SEED_NUM = 42
torch.manual_seed(SEED_NUM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config_file = yaml.safe_load(file)


class DiffusionModel:
    def __init__(self):
        self.model = None
        self.ema_model = None

    def noise_cosine_scheduling(self, diffusion_timestep: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        '''Given the diffusion steps t for a batch of images, calculate the alpha_hat defined by a cosine schedule.'''

        s = config_file['noise_cosine_schedule']['s']

        def f(t: Union[float, torch.Tensor]):
            if isinstance(t, float):
                return math.cos((t+s)/(1+s) * math.pi/2)**2
            elif isinstance(t, torch.Tensor):
                return torch.cos((t+s)/(1+s) * torch.pi/2)**2
        
        alpha_hat = f(diffusion_timestep) / f(0.)

        return alpha_hat

    def diffusion(self, image: torch.Tensor, diffusion_timestep: torch.Tensor, grid_size: Union[int, tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Add noise to a batch of images.'''
        
        batch_size = image.shape[0]

        if isinstance(grid_size, int):
            noise = torch.normal(mean=0, std=1, size=(batch_size, 2, grid_size, grid_size), device=device)
        elif isinstance(grid_size, tuple):
            noise = torch.normal(mean=0, std=1, size=(batch_size, 2, grid_size[0], grid_size[1]), device=device)

        alpha_hat = self.noise_cosine_scheduling(diffusion_timestep)
        noisy_images = image.clone()
        noisy_images[:, :2, :, :] = torch.sqrt(alpha_hat) * image[:, :2, :, :] + torch.sqrt(1-alpha_hat) * noise

        return noisy_images, noise

    def train(self, denoising_model: nn.Module, train_dataloader: DataLoader, n_epochs: int, model_path: str, ema_model_path: str) -> List:
        grid_size = config_file['grid_size']
        lr_init = config_file['training']['lr_init']
        lr_decay_rate = config_file['training']['lr_decay_rate']
        lr_decay_steps = config_file['training']['lr_decay_steps']
        EMA_coef = config_file['training']['EMA_coeff']

        denoising_model.to(device)
        self.model = denoising_model
        self.ema_model = copy.deepcopy(denoising_model)
        self.ema_model.to(device)

        for param in self.ema_model.parameters():
            param.requires_grad = False

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            params=denoising_model.parameters(), 
            lr=lr_init,
            weight_decay=config_file['training']['weight_decay']
            )
        
        n_train_batches = len(train_dataloader)
        mse_train_history = []

        for epoch in range(n_epochs):
            mse_train_loss = 0
            denoising_model.train()

            for j, x_batch in enumerate(train_dataloader):
                optimizer.zero_grad()  
                x_batch = x_batch.to(device=device)

                diffusion_timestep = torch.rand(x_batch.shape[0], 1, 1, 1, device=device)
                noisy_images, noise = self.diffusion(x_batch, diffusion_timestep, grid_size)
                noise_pred = self.model(noisy_images, diffusion_timestep)

                batch_loss = criterion(noise, noise_pred)  
                batch_loss.backward()                  
                optimizer.step()
                mse_train_loss += batch_loss.item()

                opt_step = epoch*n_train_batches + j + 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_init * lr_decay_rate ** (opt_step / lr_decay_steps)

                with torch.no_grad():
                    for model_param, ema_model_param in zip(self.model.parameters(), self.ema_model.parameters()):
                        ema_model_param.data.mul_(EMA_coef).add_(model_param.data, alpha=1 - EMA_coef)

            mse_train_loss = mse_train_loss / n_train_batches
            mse_train_history.append(mse_train_loss)

            print(f"Epoch: {epoch+1} | train MSE: {mse_train_loss:.5f}")
            
        print('\nTraining finished.\n')
        # torch.save(self.model.state_dict(), model_path)
        # torch.save(self.ema_model.state_dict(), ema_model_path)
            
        return mse_train_history
    
    def sampling(self, n_diffusion_steps: int, initial_cond: torch.Tensor) -> torch.Tensor:
        '''Generate a new field based on the given initial_cond (initial conditions + physical time).'''

        noise = torch.normal(mean=0, std=1, size=(1, 2, config_file['grid_size'], config_file['grid_size']), device=device)

        alpha_hat_list = []
        beta_list = []
        diffusion_timesteps = torch.linspace(0, 1, n_diffusion_steps+1, device=device)

        for t in range(len(diffusion_timesteps)):
            alpha_hat = self.noise_cosine_scheduling(diffusion_timesteps[t])
            alpha_hat_list.append(alpha_hat)

            if t > 0:
                beta_t = 1 - alpha_hat / (alpha_hat_list[t-1])
            else: 
                beta_t = 1 - alpha_hat

            beta_list.append(beta_t)

        model = self.ema_model
        model.eval()
        current_image = torch.cat((noise, initial_cond), dim=1)

        for i, t in enumerate(reversed(range(n_diffusion_steps)), start=1):
            alpha_hat = alpha_hat_list[-i]
            beta_t = beta_list[-i]
            alpha_t = 1 - beta_t
            diffusion_timestep = torch.full((1, 1, 1, 1), fill_value=diffusion_timesteps[-i], device=device)

            with torch.no_grad():
                noise_pred = model(current_image, diffusion_timestep)  

            if t > 0:
                z = torch.randn_like(noise, device=device)
            else:
                z = torch.zeros_like(noise)

            current_image[:, :2, :, :] = 1 / (math.sqrt(alpha_t) + 1e-8) * (current_image[:, :2, :, :] - (1-alpha_t)/math.sqrt(1-alpha_hat) * noise_pred) + math.sqrt(beta_t) * z  

        return current_image
