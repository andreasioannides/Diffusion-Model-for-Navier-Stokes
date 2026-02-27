from pathlib import Path
import os
import yaml
import torch
import torch.nn as nn
import math


SEED_NUM = 42
torch.manual_seed(SEED_NUM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Device: {device}')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config_file = yaml.safe_load(file)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, device=device)
        nn.init.xavier_normal_(self.conv1.weight)
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=out_channels, device=device)
        self.silu = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False, device=device)
            nn.init.xavier_normal_(self.shortcut.weight)  
        else:
            self.shortcut = None
    
    def forward(self, x: torch.Tensor):
        identity = x

        x = self.conv1(x)
        x = self.group_norm(x)
        x = self.silu(x)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        x = torch.add(x, identity)

        return x
    
class TimeEmbeddingMLP(nn.Module):
    '''Map the diffusion step vector to the size of the ResNet blocks' output feature maps for element-wise addition.'''

    def __init__(self, output_size: int, time_embedding_size: int = config_file['time_embedding_size']):
        super().__init__()

        self.dense1 = nn.Linear(in_features=time_embedding_size, out_features=time_embedding_size, device=device)
        nn.init.xavier_normal_(self.dense1.weight)
        self.silu = nn.SiLU()
        self.dense2 = nn.Linear(in_features=time_embedding_size, out_features=output_size, device=device)
        nn.init.xavier_normal_(self.dense2.weight)

    def forward(self, x: torch.Tensor):
        x = self.dense1(x)
        x = self.silu(x)
        x = self.dense2(x)

        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, n_dense_units: int):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True, device=device)
        self.dense1 = nn.Linear(in_features=embed_dim, out_features=n_dense_units, device=device)
        nn.init.xavier_normal_(self.dense1.weight)
        self.dense2 = nn.Linear(in_features=n_dense_units, out_features=embed_dim, device=device)
        nn.init.xavier_normal_(self.dense2.weight)
        self.silu1 = nn.SiLU()
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        identity = x.reshape(B, H*W, C)  
        x = self.layer_norm1(identity)  # normalize across the last dimension of x
        x, _ = self.multihead_attention(x, x, x)
        x = torch.add(x, identity)

        identity = x
        x = self.dense1(x)
        x = self.silu1(x)
        x = self.dense2(x)
        x = torch.add(x, identity)
        x = self.layer_norm2(x)

        x = x.reshape(B, C, H, W)

        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.resnet_block_1 = ResidualBlock(in_channels, out_channels)
        self.resnet_block_2 = ResidualBlock(out_channels, out_channels)
        self.time_emb_mlp = TimeEmbeddingMLP(output_size=out_channels)
        self.transformer_block = TransformerBlock(embed_dim=out_channels, n_heads=4, n_dense_units=4*out_channels)
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, device=device)  # downsampling
        nn.init.xavier_normal_(self.conv.weight)

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, skip_connections: list):
        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        self.time_emb = self.time_emb_mlp(time_embedding).reshape(x.shape[0], self.out_channels, 1, 1)  # [C] -> [B, C, 1, 1]

        x = torch.add(x, self.time_emb)
        x = self.transformer_block(x)
        x = self.conv(x)

        skip_connections.append(x)

        return x
    
class DecoderBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()

            self.resnet_block_1 = ResidualBlock(in_channels, out_channels)
            self.resnet_block_2 = ResidualBlock(out_channels, out_channels)
            self.time_emb_mlp = TimeEmbeddingMLP(output_size=out_channels)
            self.transformer_block = TransformerBlock(embed_dim=out_channels, n_heads=4, n_dense_units=4*out_channels)
            self.conv_transp = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, device=device)  # upsampling
            nn.init.xavier_normal_(self.conv_transp.weight)

            self.out_channels = out_channels

        def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, skip_connection: torch.Tensor):
            x = torch.cat((x, skip_connection), dim=1)

            x = self.resnet_block_1(x)
            x = self.resnet_block_2(x)
            self.time_emb = self.time_emb_mlp(time_embedding).reshape(x.shape[0], self.out_channels, 1, 1)  

            x = torch.add(x, self.time_emb)
            x = self.transformer_block(x)
            x = self.conv_transp(x)

            return x

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        half_dim = embedding_dim // 2
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half_dim))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies)

    def forward(self, x: torch.Tensor):
        embeddings = torch.cat(
            [torch.sin(x * self.angular_speeds), torch.cos(x * self.angular_speeds)],
            dim=-1
        )

        return embeddings
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=1, device=device)  # (B, C, H, W)
        nn.init.xavier_normal_(self.conv_1.weight) 
        self.noise_embedding = SinusoidalEmbedding(config_file['time_embedding_size'])

        self.encoder_block_1 = EncoderBlock(in_channels=32, out_channels=32)
        self.encoder_block_2 = EncoderBlock(in_channels=32, out_channels=64)
        self.encoder_block_3 = EncoderBlock(in_channels=64, out_channels=96)
        self.encoder_block_4 = EncoderBlock(in_channels=96, out_channels=128)

        self.resnet_block_1 = ResidualBlock(in_channels=128, out_channels=156)
        self.transformer_block = TransformerBlock(embed_dim=156, n_heads=4, n_dense_units=4*156)
        self.resnet_block_2 = ResidualBlock(in_channels=156, out_channels=156)

        self.decoder_block_1 = DecoderBlock(in_channels=156+128, out_channels=128)
        self.decoder_block_2 = DecoderBlock(in_channels=128+96, out_channels=96)
        self.decoder_block_3 = DecoderBlock(in_channels=96+64, out_channels=64)
        self.decoder_block_4 = DecoderBlock(in_channels=64+32, out_channels=32)

        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, device=device)
        nn.init.zeros_(self.conv_2.weight) 

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor):
        skip_connections = []

        noise_emb = self.noise_embedding(time_embedding)
        x = self.conv_1(x)

        x = self.encoder_block_1(x, noise_emb, skip_connections)
        x = self.encoder_block_2(x, noise_emb, skip_connections)
        x = self.encoder_block_3(x, noise_emb, skip_connections)
        x = self.encoder_block_4(x, noise_emb, skip_connections)

        x = self.resnet_block_1(x)
        x = self.transformer_block(x)
        x = self.resnet_block_2(x)

        x = self.decoder_block_1(x, noise_emb, skip_connections[3])
        x = self.decoder_block_2(x, noise_emb, skip_connections[2])
        x = self.decoder_block_3(x, noise_emb, skip_connections[1])
        x = self.decoder_block_4(x, noise_emb, skip_connections[0])

        x = self.conv_2(x)

        return x
    
