import torch
import torch.nn as nn
from tqdm import tqdm
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
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        
        # Using reparametrization trick for get noised image
        noise_image = sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*e
        return noise_image
    
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
                predicted_noise = model(x, t)
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




       
