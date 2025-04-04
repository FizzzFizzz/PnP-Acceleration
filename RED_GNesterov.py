import torch.nn as nn
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import logging
import os
from natsort import os_sorted
import sys 
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
parser.add_argument('--Nesterov', dest='Nesterov', action='store_true')
parser.set_defaults(Nesterov=False)
parser.add_argument('--momentum', dest='momentum', action='store_true')
parser.set_defaults(momentum=False)
parser.add_argument('--restarting_su', dest='restarting_su', action='store_true')
parser.set_defaults(restarting_su=False)
parser.add_argument('--r', type=int, default=3, help = "Parameter for the Generalized Nesterov momentum")
parser.add_argument('--lamb', type=float, default=18, help = "Regularization parameter")
parser.add_argument('--denoiser_level', type=float, default=0.1, help = "Denoiser level in [0.,1.]")
parser.add_argument('--sigma_obs', type=float, default=12.75, help = "Standard variation of the noise in the observation in [0.,255.]")
parser.add_argument('--dataset_name', type=str, default='set1', help = "Name of the dataset of image to restore")
parser.add_argument('--kernel_name', type=str, default='levin_6.png', help = "Name of the kernel of blur")
parser.add_argument('--stepsize', type=float, default=0.02, help = "Stepsize of the gradient descent algorithm")
parser.add_argument('--theta', type=float, default=0.9, help = "Momentum parameter")
parser.add_argument('--dont_save_images', dest='dont_save_images', action='store_true')
parser.set_defaults(dont_save_images=False)
parser.add_argument('--save_each_itr', dest='save_each_itr', action='store_true')
parser.set_defaults(save_each_itr=False)
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
        self.nb_restart_activ = 0

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

    def forward(self, initial_uv, obs, clean, kernel, sigma_obs, lamb=690, denoiser_sigma=25./255., theta = 0.9, r=3, Nesterov = False, momentum = False, restarting_su = False, stepsize = 0.02):
        '''
            Computed the RED Algorithm with
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

        y_denoised = u
        x = u
        y = u

        if restarting_su:
            # Restarting creterion proposed by "A Differential Equation for Modeling Nesterov’s Accelerated Gradient Method: Theory and Insights" section 5
            restart_crit_su = -float('inf') 
            j = 0
            k_min = 5

        for k in tqdm(range(self.nb_itr)):
            x_old = x
            self.get_psnr_i(torch.clamp(y_denoised, min = -0., max =1.), clean, k)

            data_grad = abs_k * deblur.fftn(y) - fft_kH * deblur.fftn(obs)
            data_grad = torch.real(deblur.ifftn(data_grad))

            y = y.type(torch.cuda.FloatTensor)
            y_denoised = self.net.forward(y,denoiser_sigma)
            reg_grad = y - y_denoised

            grad = reg_grad + lamb *data_grad

            x = y - stepsize*grad
            if restarting_su:
                restart_crit_su_old = restart_crit_su
                restart_crit_su = torch.mean(torch.abs(x-x_old)).item()
                if (k>k_min) and (restart_crit_su<restart_crit_su_old):
                    j = 0
                    self.nb_restart_activ += 1
                else:
                    j += 1
            else:
                j = k

            if Nesterov:
                if k+r==0:
                    y = x + (x-x_old)
                else:
                    y = x + j/(j+r)*(x-x_old)
            elif momentum:
                y = x + (1-theta)*(x - x_old)
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


# Parameters setting
denoiser_level = hparams.denoiser_level
lamb = hparams.lamb
sigma_obs = hparams.sigma_obs
r = hparams.r
theta = hparams.theta
Nesterov = hparams.Nesterov
momentum = hparams.momentum
restarting_su = hparams.restarting_su
stepsize = hparams.stepsize
dont_save_images = hparams.dont_save_images
save_each_itr = hparams.save_each_itr

model = PnP()
model.to(device)
model.net.to(device)
model.eval()
model.net.eval()

# Set input image paths
input_path = os.path.join('datasets', hparams.dataset_name)
input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

for i, clean_image_path in enumerate(input_paths):
    kernel_path = 'utils/kernels/'+hparams.kernel_name
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

    # Run RED algorithm with or without Nesterov momentum
    with torch.no_grad():
        model(initial_uv, observation, clean_image, kernel, sigma_obs, lamb, denoiser_level, theta, r, Nesterov, momentum, restarting_su, stepsize)

    psnr_list = model.res['psnr']
    ssim_list = model.res['ssim']
    print("Restored image PSNR = {:.2f}".format(psnr_list[-1]))
    if restarting_su:
        print("Number of restarting activation = {}".format(model.nb_restart_activ))
    

    savepth = 'results/'+hparams.dataset_name+"/RED_level_{}_lamb{}".format(denoiser_level, lamb)
    if Nesterov:
        savepth = savepth + "_Nesterov_r_{}".format(r)
    elif momentum:
        savepth = savepth + "_Heavy_ball_theta_{}".format(theta)
    if restarting_su:
        savepth = savepth + "restarting_su"
    os.makedirs(savepth, exist_ok = True)

    if not(dont_save_images):
        if save_each_itr:
            savepth_img = savepth+"/set_img_{}/".format(i)
            os.makedirs(savepth_img, exist_ok = True)
            for j in range(len(model.res['image'])):
                model.res['image'][j].save(savepth_img + 'iterations_{}.png'.format(j))
        
        model.res['image'][-1].save(savepth + '/restored_img.png')
        clean_img_uint = util.tensor2uint(clean_image)
        obs_uint = util.tensor2uint(observation)
        plt.imsave(savepth + '/clean_img.png', clean_img_uint)
        plt.imsave(savepth + '/observation.png', obs_uint)

        itr_list = range(len(psnr_list))
        plt.clf()
        plt.plot(itr_list, psnr_list, '-', alpha=0.8, linewidth=1.5)
        plt.xlabel('iter')
        plt.ylabel('PSNR')
        plt.savefig(savepth+'/PSNR_list.png')
        plt.clf()
        plt.plot(itr_list, ssim_list, '-', alpha=0.8, linewidth=1.5)
        plt.xlabel('iter')
        plt.ylabel('SSIM')
        plt.savefig(savepth+'/SSIM_list.png')
    
    dict = {
            'clean_image' : util.tensor2uint(clean_image),
            'observation' : util.tensor2uint(observation),
            'initial_uv' : initial_uv,
            'kernel' : util.tensor2uint(kernel),
            'sigma_obs' : sigma_obs,
            'lamb' : lamb,
            'denoiser_level' : denoiser_level,
            'r' : r,
            'Nesterov' : Nesterov,
            'restarting_su' : restarting_su,
            'stack_images' : model.res['image'],
            'clean_image_path' : clean_image_path,
            'kernel_path' : kernel_path,
            'psnr_list' : psnr_list,
            'ssim_list' : ssim_list,
            'psnr_restored' : psnr_list[-1],
            'ssim_restored' : ssim_list[-1],
            'restored' : model.res['image'][-1],
            'nb_restart_activ' : model.nb_restart_activ,
        }
    
    np.save(savepth+"/dict_results_{}".format(i), dict)
