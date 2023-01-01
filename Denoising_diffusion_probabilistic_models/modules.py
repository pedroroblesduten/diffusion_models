import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init(self, channels, size):
        super(SelfAttention, self).__init()

        self.channels = channels
        self.size = size
        self.multi_head_attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm([channels])
        self.feed_foward = nn.Sequential(
            nn.layer_norm([channels]),
            nn.Linear(channels, channels)
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1, 2)
        x = self.layer_norm(x)
        h, _ = self.multi_head_attention(x, x, x)
        x = h + x
        h = self.feed_foward(x)
        x = h + x
        x = x.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None, residual=True):
        super().__init()
        
        self.residual = residual
        if not mid_c:
            mid_c = out_c

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=False),
            nn.GroupNorm(1, mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_c)
        )
    def forward(self, x):
        h = self.double_conv(x)
        if self.residual:
            x = F.GELU(x + h)
        else:
            x = h
        return x

class DownSampleBlock(nn.Module):
    def __init(self, in_c, out_c, emb_dim)
        super().__init()

        self.block1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_c, out_c, residual=False),
            ConvBlock(in_c, out_c, residual=True)
        )
        self.embedding = nn.Sequential](
            nn.SiLU(),
            nn.Linear(emb_dim, out_c)
        )

    def forward(self, x, t):
        x = self.block1(x)
        emb = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UpSampleBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init()
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear'),
                                    align_corners=True)
        self.block1 = nn.Sequential(
            ConvBlock(in_c, in_c, residual=True),
            ConvBlock(in_c, out_c, in_c//2)
        )
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_c)
        )

    def forward(self, x, skip_x, t):    
            x = self.upsample(x)
            x = torch.cat([skip_x, x], dim=1)
            x = self.block1(x)
            emb = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            return x + emb



        
