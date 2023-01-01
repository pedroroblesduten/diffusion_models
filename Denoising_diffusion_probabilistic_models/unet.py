import torch
import torch.nn as nn
from modules import SelfAttention, ConvBlock, UpSampleBlock, DownSampleBlock

class Enconder(nn.Module):
    def __init__(self, in_c=3, out_c=3, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.conv1 = ConvBlock(in_c, 64, residual=False)
        self.down1 = DownSampleBlock(64, 128)
        self.self_attention1 = SelfAttention(128, 32)
        self.down2 = DownSampleBlock(128, 256)
        self.self_attention2 = SelfAttention(256, 16)
        self.down3 = DownSampleBlock(256, 256)
        self.self_attention3 = SelfAttention(256, 8)

    def forward(self, x, t):
        if self.verbose:
            print('-- STARTING ENCODER --')
            print(f'Shape original: {x.shape}')
        x1 = self.conv1(x)
        if self.verbose:
            print(f'Shape after conv1: {x1.shape}')
        x2 = self.down1(x1, t)
        if self.verbose:
            print(f'Shape after down1: {x2.shape}')
        x2 = self.self_attention1(x2)        
        if self.verbose:
            print(f'Shape after self_attention1: {x2.shape}')
        x3 = self.down2(x2, t)        
        if self.verbose:
            print(f'Shape after down2: {x3.shape}')
        x3 = self.self_attention2(x3)        
        if self.verbose:
            print(f'Shape after self_attention2: {x3.shape}')
        x4 = self.down3(x3, t)        
        if self.verbose:
            print(f'Shape after down3: {x4.shape}')
        x4 = self.self_attention3(x4)        
        if self.verbose:
            print(f'Shape after self_attention3: {x4.shape}')
        return x1, x2, x3, x4

class BottleNeck(nn.Module):
    def __init__(self, in_c=256, out_c=256, mid_c=512, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.bot1 = ConvBlock(in_c, mid_c, residual=False)
        self.bot2 = ConvBlock(mid_c, mid_c, residual=False)
        self.bot3 = ConvBlock(mid_c, out_c, residual=False)

    def forward(self, x):
        if self.verbose:
            print('-- STARTING BOTTLENECK --')
        x = self.bot1(x)        
        if self.verbose:
            print(f'Shape after conv1: {x.shape}')
        x = self.bot2(x)        
        if self.verbose:
            print(f'Shape after conv2: {x.shape}')
        x = self.bot3(x)        
        if self.verbose:
            print(f'Shape after conv3: {x.shape}')
        return x

class Decoder(nn.Module):
    def __init__(self, out_c=3, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.up1 = UpSampleBlock(512, 128)
        self.self_att1 = SelfAttention(128, 16)
        self.up2 = UpSampleBlock(256, 64)
        self.self_att2 = SelfAttention(64, 32)
        self.up3 = UpSampleBlock(128, 64)
        self.self_att3 = SelfAttention(64, 64)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x1, x2, x3, x4, t):
        if self.verbose:
            print('-- STARTING DECODER --')
        x = self.up1(x4, x3, t)        
        if self.verbose:
            print(f'Shape after up1: {x.shape}')
        x = self.self_att1(x)        
        if self.verbose:
            print(f'Shape after sel: {x.shape}')
        x = self.up2(x, x2, t)        
        if self.verbose:
            print(f'Shape after up2: {x.shape}')
        x = self.self_att2(x)        
        if self.verbose:
            print(f'Shape after self_att2: {x.shape}')
        x = self.up3(x, x1, t)        
        if self.verbose:
            print(f'Shape after up3: {x.shape}')
        x = self.self_att3(x)        
        if self.verbose:
            print(f'Shape after self_att3: {x.shape}')
        x = self.conv1(x)        
        if self.verbose:
            print(f'Shape after conv1: {x.shape}')
        return x


class UNet(nn.Module):
    def __init__(self, time_dim=256, num_classes=None, device='cuda', verbose=False):
        super().__init__()
        self.verbose = verbose
        self.device = device
        self.time_dim = time_dim
        self.encoder = Enconder(verbose=self.verbose)
        self.bottle_neck = BottleNeck(verbose=self.verbose)
        self.decoder = Decoder(verbose=self.verbose)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0/(
                10000**torch.arange(
                    0, channels, 2, device=self.device).float()/channels)
        pos_enc_a = torch.sin(t.repeat(1, channels//2)*inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2)*inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y=None):
        if self.verbose:
            print('-> Starting UNET <-')
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            t += self.label_emb(y)

        x1, x2, x3, x4 = self.encoder(x, t)
        x4 = self.bottle_neck(x4)
        output = self.decoder(x1, x2, x3, x4, t)
        if self.verbose:
            print('-> Finished UNET <-')
        return output




if __name__ == '__main__':
    net = UNet(num_classes=10, device='cpu', verbose=True)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500]*x.shape[0]).long()
    y = x.new_tensor([1]*x.shape[0]).long()
    print(net(x, t, y).shape)
