# src/diffusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block2D(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=(1, 3), padding=(0, 1))
        self.norm = nn.GroupNorm(min(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        
        if scale_shift is not None:
            scale, shift = scale_shift
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            x = x * (scale + 1) + shift
            
        x = self.act(x)
        return x

class ResnetBlock2D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = Block2D(dim, dim_out, groups=min(groups, dim, dim_out))
        self.block2 = Block2D(dim_out, dim_out, groups=min(groups, dim_out))
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=(1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        
        q = q * self.scale
        
        sim = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)

class InputAdapter(nn.Module):
    """Adapts input features to a fixed channel size for the diffusion model"""
    def __init__(self, target_channels=26):
        super().__init__()
        self.target_channels = target_channels
        self.adapters = {}
        
    def forward(self, x):
        # Get input shape
        batch_size, in_channels, height, width = x.shape
        
        # Check if we need to create a new adapter
        if in_channels not in self.adapters:
            self.adapters[in_channels] = nn.Conv2d(
                in_channels, 
                self.target_channels, 
                kernel_size=(1, 1)
            ).to(x.device)
            
            # Initialize weights properly
            nn.init.kaiming_normal_(self.adapters[in_channels].weight)
            nn.init.zeros_(self.adapters[in_channels].bias)
            
        # Apply the adapter
        return self.adapters[in_channels](x)

class OutputAdapter(nn.Module):
    """Adapts output features back to the original input channel size"""
    def __init__(self, fixed_channels=26):
        super().__init__()
        self.fixed_channels = fixed_channels
        self.adapters = {}
        
    def forward(self, x, original_channels):
        # Get input shape
        batch_size, _, height, width = x.shape
        
        # Check if we need to create a new adapter
        if original_channels not in self.adapters:
            self.adapters[original_channels] = nn.Conv2d(
                self.fixed_channels, 
                original_channels, 
                kernel_size=(1, 1)
            ).to(x.device)
            
            # Initialize weights properly
            nn.init.kaiming_normal_(self.adapters[original_channels].weight)
            nn.init.zeros_(self.adapters[original_channels].bias)
            
        # Apply the adapter
        return self.adapters[original_channels](x)

class SignDiffusionModel(nn.Module):
    def __init__(
        self,
        fixed_channels=26,  # Fixed internal channel dimension
        text_embed_dim=384,  # Default for sentence-transformers
        time_dim=256,
        dim=256,
        dim_mults=(1, 2),  # Reduced number of layers
        groups=8
    ):
        super().__init__()
        
        # Input and output adapters
        self.input_adapter = InputAdapter(fixed_channels)
        self.output_adapter = OutputAdapter(fixed_channels)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Text conditioning
        self.text_proj = nn.Linear(text_embed_dim, time_dim)
        
        # Initial projection
        self.init_conv = nn.Conv2d(fixed_channels, dim, kernel_size=(1, 3), padding=(0, 1))
        
        # Downsampling and upsampling dimensions
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Downsampling blocks
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResnetBlock2D(dim_in, dim_out, time_emb_dim=time_dim, groups=min(groups, dim_in, dim_out)),
                ResnetBlock2D(dim_out, dim_out, time_emb_dim=time_dim, groups=min(groups, dim_out)),
                Attention2D(dim_out),
                nn.Conv2d(dim_out, dim_out, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)) if ind < len(in_out) - 1 else nn.Identity()
            ]))
        
        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock2D(mid_dim, mid_dim, time_emb_dim=time_dim, groups=min(groups, mid_dim))
        self.mid_attn = Attention2D(mid_dim)
        self.mid_block2 = ResnetBlock2D(mid_dim, mid_dim, time_emb_dim=time_dim, groups=min(groups, mid_dim))
        
        # Upsampling blocks
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                ResnetBlock2D(dim_out * 2, dim_in, time_emb_dim=time_dim, groups=min(groups, dim_in, dim_out * 2)),
                ResnetBlock2D(dim_in, dim_in, time_emb_dim=time_dim, groups=min(groups, dim_in)),
                Attention2D(dim_in),
                nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), output_padding=0) if ind < len(in_out) - 1 else nn.Identity()
            ]))
        
        # Final output
        self.final_res_block = ResnetBlock2D(dim * 2, dim, time_emb_dim=time_dim, groups=min(groups, dim * 2, dim))
        self.final_conv = nn.Conv2d(dim, fixed_channels, kernel_size=(1, 1))
        
        # Fixed channels for reference
        self.fixed_channels = fixed_channels

    def forward(self, x, time, text_embed):
        # Store original input shape and channels
        input_shape = x.shape
        original_channels = x.shape[1]
        
        # Print input shape for debugging
        print(f"Input shape to forward: {x.shape}")
        
        # Adapt input to fixed channel size
        x = self.input_adapter(x)
        print(f"After adapter: {x.shape}")
        
        # Time embedding
        t = self.time_mlp(time)
        
        # Text conditioning
        c = self.text_proj(text_embed)
        t = t + c
        
        # Initial projection
        x = self.init_conv(x)
        print(f"After initial conv: {x.shape}")
        h = [x]
        
        # Downsampling
        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # Upsampling
        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)
        
        # Final
        x = torch.cat((x, h.pop()), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        # Adapt output back to original channel size
        x = self.output_adapter(x, original_channels)
        
        # Ensure output has same shape as input
        if x.shape[2:] != input_shape[2:]:
            x = F.interpolate(x, size=input_shape[2:], mode='bilinear', align_corners=False)
        
        print(f"Final output shape: {x.shape}")
        return x
