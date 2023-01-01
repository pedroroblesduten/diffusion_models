import torch
import torch.nn as nn
from tqdm import tqdm
from unet import UNet
from utils import get_data, setup_logging, save_images
import torch.optim as optim
# import logging

#Following the implemenation from: https://github.com/dome272/Diffusion-Models-pytorch
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        #Definition for DDPM paper
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    #Initializing beta for every t in steps
    def prepare_noise_schedule(self):
        # torch.linspace creates a one-dimesional tensor of size 'steps' whose
        # values are evenly sapced from start' to 'end'
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        
        # Using reparametrization trick for get noised image
        noise_image = sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*e
        return noise_image, e
    
    def sample_t(self, n):
        t = torch.randint(low=1, high=self.noise_steps, size=(n,))
        return t

    def sample(self, model, n):
        # Algorithm 2 from the DDPM paper
        model.eval()
        with torch.nn_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) #Initializing X_T from normal distribution
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n)*i).long().to(device)
                predicted_noise = model(x, t, verbose=True)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                x = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/(torch.sqrt(1-alpha_hat))) * predicted_noise)+torch.sqrt(beta)*z
        model.train()
        x = (x.clamp(-1, 1)+1)/2
        x = (x*255).type(torch.uint8)
        return x



def train(args):
    device = args.device
    dataloader = get_data(args)
    model = UNet(verbose=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    l = len(dataloader)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_t(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
             
            pbar.set_postfix(MSE=loss.item())
        sampled_images = diffusion.sample(model, n=image.shape[0])
        save_images(sampled_images, os.path.join('results', args.run_name, f'{epoch}.jpg'))
        torch.save(model.state_dict(), os.path.join('models', args.run_name, f'ckpt.pt'))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = r"C:\Users\pedro\OneDrive\Área de Trabalho\diffusion_images"
    args.epochs = 100
    args.batch_size = 2
    args.image_size = 64
    args.dataset_path = r"C:\Users\pedro\OneDrive\Área de Trabalho\flowers"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()
