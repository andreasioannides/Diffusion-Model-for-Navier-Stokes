from pathlib import Path
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from typing import Union, Tuple, List


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

    def factors_cosine_schedule(self, diffusion_timesteps: torch.Tensor) -> tuple:
        '''Define a codine schedule for the image and noise factors. (img_factor^2 + noise_factor^2 = 1)'''

        image_factor_min = config_file['cosine_schedule']['image_factor_min']
        image_factor_max = config_file['cosine_schedule']['image_factor_max']

        start_angle = torch.acos(torch.tensor(image_factor_max, device=device))
        end_angle = torch.acos(torch.tensor(image_factor_min, device=device))
        diffusion_angles = start_angle + diffusion_timesteps * (end_angle - start_angle)

        image_factors = torch.cos(diffusion_angles)
        noise_factors = torch.sin(diffusion_angles)

        return image_factors, noise_factors

    def diffusion(self, images: torch.Tensor, grid_size: Union[int, tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Add noise to the images.'''
        
        batch_size = images.shape[0]

        if isinstance(grid_size, int):
            noise = torch.normal(mean=0, std=1, size=(batch_size, 2, grid_size, grid_size), device=device)
        elif isinstance(grid_size, tuple):
            noise = torch.normal(mean=0, std=1, size=(batch_size, 2, grid_size[0], grid_size[1]), device=device)

        diffusion_timesteps = torch.rand(batch_size, 1, 1, 1, device=device)
        image_factors, noise_factors = self.factors_cosine_schedule(diffusion_timesteps)
        noisy_images = images.clone()
        noisy_images[:, :2, :, :] = image_factors[:, :2, :, :] * images[:, :2, :, :] + noise_factors * noise

        return noisy_images, noise, image_factors, noise_factors
    
    def reverse_diffusion(self, noisy_images: torch.Tensor, image_factors: torch.Tensor, noise_factors: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Predict the added noise to the image x_t and denoise the image.'''

        if training:
            model = self.model
            noise_pred = model(noisy_images, noise_factors**2)
        else:
            model = self.ema_model
            model.eval()

            with torch.no_grad():
                noise_pred = model(noisy_images, noise_factors**2)

        images_pred = (noisy_images[:, :2, :, :] - noise_factors * noise_pred) / image_factors  # ommit the channels for initial conditions and physical time

        return images_pred, noise_pred

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

                noisy_images, noise, image_factors, noise_factors = self.diffusion(x_batch, grid_size)
                images_pred, noise_pred = self.reverse_diffusion(noisy_images, image_factors, noise_factors, training=True) 

                batch_loss = criterion(noise, noise_pred) / n_train_batches  
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

            print(f"Epoch: {epoch+1}, train MSE: {mse_train_loss:.5f}")
            
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.ema_model.state_dict(), ema_model_path)
            
        return mse_train_history
    
    # def sampling(self):
