import torch.nn as nn
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import logging
import os
import sys 
import cv2
from PIL import Image
# sys.path.append("..")
sys.path.append("utils/")
import utils_image as util
import utils_logger
import utils_deblur as deblur
sys.path.append("utils/models/") 
from models.network_unet import UNetRes as Net
from tqdm import tqdm


    
model_path = "models_ckpt/DRUNet_color.pth"
n_channels = 3
device = 'cuda:0'

# model.eval()
# model = model.cuda()


class Drunet_running(torch.nn.Module):# DRUNet model definition 
    def __init__(self, model_path, n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose", bias=False):
        super(Drunet_running, self).__init__()
        self.model = Net(in_nc=n_channels+1, out_nc=n_channels, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)    

    def forward(self, x, sigma):
        '''
        x : image with values in [0, 1]
        sigma : standard deviation of denoising in [0, 1]
        '''
        sigma = float(sigma)
        sigma_div_255 = torch.FloatTensor([sigma]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
        x = torch.cat((x, sigma_div_255), dim=1)
        return self.model(x)



class PnP(nn.Module):
    def __init__(self, nb_itr=50):
        '''
            nb_itr : number of iterations of the PnP algorithm
        '''
        super(PnP, self).__init__()
        self.nb_itr = nb_itr
        self.net = Drunet_running(model_path = model_path, n_channels = n_channels)

        # only test
        self.res = {}
        self.res['psnr'] = [0] * nb_itr
        self.res['ssim'] = [0] * nb_itr
        self.res['image'] = [0]* nb_itr

    # def IRL1(self, f, u, v, b2, sigma, lamb, sigma2, k=10, eps=1e-5):
    #     for j in range(k):
    #         fenzi = lamb * (v-f)/(sigma**2+(v-f)**2)+(v-u-b2)
    #         fenmu = lamb * (sigma**2-(v-f)**2)/(sigma**2+(v-f)**2)**2+1
    #         v = v - fenzi / fenmu
    #         v = torch.clamp(v, min=0, max=255.)
    #     return v

    def get_psnr_i(self, u, clean, i):
        '''
        Compute the PSNR, SSIM and save the image at the iteration i.
        '''
        psnr = util.calculate_psnr_torch(u / 255., clean).item()
        self.res['psnr'][i] = psnr
        ssim = util.calculate_ssim_torch(u / 255., clean).item()
        self.res['ssim'][i] = ssim
        pre_i = torch.clamp(u / 255., 0., 1.)
        self.res['image'][i] = ToPILImage()(pre_i[0])

    def forward(self, kernel, initial_uv, obs, clean, sigma_obs=25.5, lamb=690, sigma2=1.0, denoisor_sigma=25, r=0): 
        # init
        obs *= 255
        u  = obs
        average = obs
        K = kernel

        fft_k = deblur.p2o(K, u.shape[-2:])
        fft_kH = torch.conj(fft_k)
        abs_k = fft_kH * fft_k

        lamb_ = lamb
        d = denoisor_sigma
        t = u
        w = u
        x = u 
        y = u

        for k in tqdm(range(self.nb_itr)):
            oldx = x
            self.get_psnr_i(torch.clamp(w, min = -0., max =255.), clean, k)

            temp = abs_k * deblur.fftn(y) - fft_kH * deblur.fftn(obs)
            temp = torch.real(deblur.ifftn(temp))

            t = y/255
            t = t.type(torch.cuda.FloatTensor)
            w = self.net.forward(t,d) * 255

            G =  y - w + lamb_*( temp ) 

            x = y - 0.02*G
            if k+r==0:
                y = x + (x-oldx)
            else:
                y = x + (k)/(k+r)*(x-oldx)
        return w

def gen_data(clean_image, sigma, kernel, seed=0):
    """
    Generate the degradate observation
    """
    fft_k = deblur.p2o(kernel, clean_image.shape[-2:])
    temp = fft_k * deblur.fftn(clean_image)
    observation_without_noise = torch.abs(deblur.ifftn(temp))

    # For reproducibility
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise = torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen)*sigma / 255

    observation = observation_without_noise + noise
    return observation

def plot_psnr(denoiser_level, lamb, sigma_obs, r):
    model = PnP()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()
    
    sigma2 = 1.0

    clean_image_path = 'CBSD68_cut8/0004.png'
    kernel_path = 'utils/kernels/levin_6.png'
    kernel = util.imread_uint(kernel_path,1)
    kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
    kernel = kernel / torch.sum(kernel)
    clean_image = util.imread_uint(clean_image_path, 3)
    clean_image = util.single2tensor3(clean_image).unsqueeze(0) /255.
    observation = gen_data(clean_image, sigma_obs, kernel)

    observation = observation.to(device)
    initial_uv = observation.clone()
    clean_image = clean_image.to(device)
    kernel = kernel.to(device)

    with torch.no_grad():
        model(kernel, initial_uv, observation, clean_image, sigma_obs, lamb, sigma2, denoiser_level, r)

    savepth = 'images_GNesterov_RED_r{}'.format(r)+'/'
    for j in range(len(model.res['image'])):
        # model.res['image'][j].save(savepth + 'result_Brain{}_{}.png'.format(i, j))
        model.res['image'][j].save(savepth + 'result_{}.png'.format(j))

    y = model.res['psnr']
    print("Restored image PSNR = {:.2f}".format(y[-1]))
    x = range(len(y))
    plt.plot(x, y, '-', alpha=0.8, linewidth=1.5)
    plt.xlabel('iter')
    plt.ylabel('PSNR')
    plt.savefig('PSNR_level{}_lamb{}_r{}_RED_GeneralizedNesterov.png'.format(denoiser_level, lamb,r))






## RED 
plot_psnr(denoiser_level = 0.1, lamb = 18, sigma_obs = 12.75, r = 4)
