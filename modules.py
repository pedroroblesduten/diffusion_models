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

class DoubleConv(nn.Module):
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



        
