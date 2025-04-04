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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, default="models_ckpt/DRUNet_color.pth", help = "The path for the DRUNet pretrained weights")
parser.add_argument('--n_channels', type=int, default=3, help = "number of channels of the image, by default RGB")
parser.add_argument('--gpu_number', type=int, default=0, help = "the GPU number")
parser.add_argument('--no_momentum', dest='no_momentum', action='store_true')
parser.set_defaults(early_stopping=False)
parser.add_argument('--r', type=int, default=3, help = "Parameter for the Generalized Nesterov momentum")
parser.add_argument('--lamb', type=float, default=18, help = "Regularization parameter")
parser.add_argument('--denoiser_level', type=float, default=0.1, help = "Denoiser level in [0.,1.]")
parser.add_argument('--sigma_obs', type=float, default=12.75, help = "Standard variation of the noise in the observation in [0.,255.]")
hparams = parser.parse_args()

model_path = hparams.model_path
n_channels = 3
device = torch.device('cuda:'+str(hparams.gpu_number) if torch.cuda.is_available() else 'cpu')

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

    def get_psnr_i(self, u, clean, i):
        '''
        Compute the PSNR, SSIM and save the image at the iteration i.
        '''
        psnr = util.calculate_psnr_torch(u, clean).item()
        self.res['psnr'][i] = psnr
        ssim = util.calculate_ssim_torch(u, clean).item()
        self.res['ssim'][i] = ssim
        pre_i = torch.clamp(u, 0., 1.)
        self.res['image'][i] = ToPILImage()(pre_i[0])

    def forward(self, initial_uv, obs, clean, kernel, sigma_obs, lamb=690, denoiser_sigma=25./255., r=3, momentum = True, stepsize = 0.02):
        '''
            Computed the Generalized Nesterov Algorithm with
                initial_uv : the initialization for the algorithm
                obs : the observation, degraded image
                clean : the clean image to compute the metrics at each iterations
                kernel : the kernel of blur
                sigma_obs : the noise level of the observation
                lamb : the regularization parameter, multiplicative factor in front of the data-fidelity
                denoiser_sigma : the noise parameter of the denoiser
                r : the parameter of the Generalized Nesterov momentum, r = 3, we recover the classic Nesterov momentum
                stepsize : the stepsize of the algorithm
        '''
        # init
        u  = obs
        # average = obs
        K = kernel
        fft_k = deblur.p2o(K, u.shape[-2:])
        fft_kH = torch.conj(fft_k)
        abs_k = fft_kH * fft_k

        t = u
        y_denoised = u
        x = u
        y = u

        for k in tqdm(range(self.nb_itr)):
            oldx = x
            self.get_psnr_i(torch.clamp(y_denoised, min = -0., max =1.), clean, k)

            data_grad = abs_k * deblur.fftn(y) - fft_kH * deblur.fftn(obs)
            data_grad = torch.real(deblur.ifftn(data_grad))

            y = y.type(torch.cuda.FloatTensor)
            y_denoised = self.net.forward(y,denoiser_sigma)
            reg_grad = y - y_denoised

            grad = reg_grad + lamb *data_grad

            x = y - stepsize*grad
            if momentum:
                if k+r==0:
                    y = x + (x-oldx)
                else:
                    y = x + (k)/(k+r)*(x-oldx)
            else:
                y = x
        return y_denoised

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

def plot_psnr(denoiser_level, lamb, sigma_obs, r, momentum):
    model = PnP()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()

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
        model(initial_uv, observation, clean_image, kernel, sigma_obs, lamb, denoiser_level, r, momentum)

    if momentum:
        savepth = 'images_GNesterov_RED_r{}'.format(r)+'/'
    else:
        savepth = 'images_RED/'
    for j in range(len(model.res['image'])):
        # model.res['image'][j].save(savepth + 'result_Brain{}_{}.png'.format(i, j))
        model.res['image'][j].save(savepth + 'result_{}.png'.format(j))

    y = model.res['psnr']
    print("Restored image PSNR = {:.2f}".format(y[-1]))
    x = range(len(y))
    plt.plot(x, y, '-', alpha=0.8, linewidth=1.5)
    plt.xlabel('iter')
    plt.ylabel('PSNR')
    if momentum:
        plt.savefig('PSNR_level_{}_lamb{}_r{}_RED_GeneralizedNesterov.png'.format(denoiser_level, lamb,r))
    else:
        plt.savefig('PSNR_level_{}_lamb{}_RED.png'.format(denoiser_level, lamb,r))


# Run RED algorithm with or without Nesterov momentum
plot_psnr(denoiser_level = hparams.denoiser_level, lamb = hparams.lamb, sigma_obs = hparams.sigma_obs, r = hparams.r, momentum = not(hparams.no_momentum))
