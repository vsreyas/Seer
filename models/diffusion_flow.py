import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np
import time
__version__ = "0.0"
from contextlib import suppress

import os

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

import tensorboard as tb

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusionSeerFlow(nn.Module):
    def __init__(
        self,
        model,
        args,
        *,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.args = args
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        t,
        image_primary,
        image_wrist,
        state,
        text_token,
        noisy_imgs=None,
        noisy_actions=None,
        guidance_weight=0,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        """
        Forward pass through SeerAgentFlow and apply diffusion objective logic
        Returns:
            dict with keys "image" and "action", each a ModelPrediction
        """

        # Forward through SeerAgentFlow
        arm_pred_action, gripper_pred_action, image_pred, *_ = self.model(
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            noisy_imgs=noisy_imgs,
            noisy_actions=noisy_actions,
            t=t,
        )

        # classifier-free guidance (optional, set guidance_weight > 0)
        if guidance_weight > 0:
            uncond_out = self.model(
                image_primary=image_primary,
                image_wrist=image_wrist,
                state=state,
                text_token=text_token,
                noisy_imgs=noisy_imgs,
                noisy_actions=noisy_actions,
                t=t,
            )
            # TODO: zero out text_token for unconditional (like in CFG)
            arm_pred_action_u, gripper_pred_action_u, image_pred_u, *_ = uncond_out
        else:
            arm_pred_action_u = gripper_pred_action_u = image_pred_u = None

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        out = {}

        # -------- Image branch --------
        if image_pred is not None and noisy_imgs is not None:
            if self.objective == "pred_noise":
                pred_noise = image_pred if guidance_weight == 0 else \
                    (1 + guidance_weight) * image_pred - guidance_weight * image_pred_u
                x_start = self.predict_start_from_noise(noisy_imgs, t, pred_noise)
                x_start = maybe_clip(x_start)
                if clip_x_start and rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(noisy_imgs, t, x_start)

            elif self.objective == "pred_x0":
                x_start = maybe_clip(image_pred)
                if guidance_weight == 0:
                    pred_noise = self.predict_noise_from_start(noisy_imgs, t, x_start)
                else:
                    cond_noise = self.predict_noise_from_start(noisy_imgs, t, x_start)
                    uncond_noise = self.predict_noise_from_start(noisy_imgs, t, maybe_clip(image_pred_u))
                    pred_noise = (1 + guidance_weight) * cond_noise - guidance_weight * uncond_noise
                    x_start = self.predict_start_from_noise(noisy_imgs, t, pred_noise)

            elif self.objective == "pred_v":
                v = image_pred
                x_start = maybe_clip(self.predict_start_from_v(noisy_imgs, t, v))
                if guidance_weight == 0:
                    pred_noise = self.predict_noise_from_start(noisy_imgs, t, x_start)
                else:
                    cond_noise = self.predict_noise_from_start(noisy_imgs, t, x_start)
                    uncond_x_start = self.predict_start_from_v(noisy_imgs, t, image_pred_u)
                    uncond_noise = self.predict_noise_from_start(noisy_imgs, t, uncond_x_start)
                    pred_noise = (1 + guidance_weight) * cond_noise - guidance_weight * uncond_noise
                    x_start = self.predict_start_from_noise(noisy_imgs, t, pred_noise)

            out["image"] = ModelPrediction(pred_noise, x_start)

        # -------- Action branch --------
        if arm_pred_action is not None and gripper_pred_action is not None and noisy_actions is not None:
            action_pred = torch.cat([arm_pred_action, gripper_pred_action], dim=-1)  # (B, horizon, 7)
            if guidance_weight > 0:
                action_pred_u = torch.cat([arm_pred_action_u, gripper_pred_action_u], dim=-1)

            if self.objective == "pred_noise":
                pred_noise = action_pred if guidance_weight == 0 else \
                    (1 + guidance_weight) * action_pred - guidance_weight * action_pred_u
                x_start = self.predict_start_from_noise(noisy_actions, t, pred_noise)
                x_start = maybe_clip(x_start)
                if clip_x_start and rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(noisy_actions, t, x_start)

            elif self.objective == "pred_x0":
                x_start = maybe_clip(action_pred)
                if guidance_weight == 0:
                    pred_noise = self.predict_noise_from_start(noisy_actions, t, x_start)
                else:
                    cond_noise = self.predict_noise_from_start(noisy_actions, t, x_start)
                    uncond_noise = self.predict_noise_from_start(noisy_actions, t, maybe_clip(action_pred_u))
                    pred_noise = (1 + guidance_weight) * cond_noise - guidance_weight * uncond_noise
                    x_start = self.predict_start_from_noise(noisy_actions, t, pred_noise)

            elif self.objective == "pred_v":
                v = action_pred
                x_start = maybe_clip(self.predict_start_from_v(noisy_actions, t, v))
                if guidance_weight == 0:
                    pred_noise = self.predict_noise_from_start(noisy_actions, t, x_start)
                else:
                    cond_noise = self.predict_noise_from_start(noisy_actions, t, x_start)
                    uncond_x_start = self.predict_start_from_v(noisy_actions, t, action_pred_u)
                    uncond_noise = self.predict_noise_from_start(noisy_actions, t, uncond_x_start)
                    pred_noise = (1 + guidance_weight) * cond_noise - guidance_weight * uncond_noise
                    x_start = self.predict_start_from_noise(noisy_actions, t, pred_noise)

            out["action"] = ModelPrediction(pred_noise, x_start)

        return out


    # def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised=False, guidance_weight=0):
    #     preds = self.model_predictions(x, t, x_cond, task_embed, guidance_weight=guidance_weight)
    #     x_start = preds.pred_x_start

    #     if clip_denoised:
    #         x_start.clamp_(-1., 1.)

    #     model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
    #     return model_mean, posterior_variance, posterior_log_variance, x_start

    # @torch.no_grad()
    # def p_sample(self, x, t: int, x_cond, task_embed, guidance_weight=0):
    #     b, *_, device = *x.shape, x.device
    #     batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
    #     model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed, clip_denoised = True, guidance_weight=guidance_weight)
    #     noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
    #     pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    #     return pred_img, x_start

    # @torch.no_grad()
    # def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
    #     batch, device = shape[0], self.betas.device

    #     img = torch.randn(shape, device=device)
    #     imgs = [img]

    #     x_start = None

    #     for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
    #         # self_cond = x_start if self.self_condition else None
    #         img, x_start = self.p_sample(img, t, x_cond, task_embed, guidance_weight=guidance_weight)
    #         imgs.append(img)

    #     ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    #     ret = self.unnormalize(ret)
    #     return ret

    def p_mean_variance(
        self,
        noisy_imgs,
        noisy_actions,
        t,
        image_primary,
        image_wrist,
        state,
        text_token,
        clip_denoised=False,
        guidance_weight=0,
    ):
        preds = self.model_predictions(
            t=t,
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            noisy_imgs=noisy_imgs,
            noisy_actions=noisy_actions,
            guidance_weight=guidance_weight,
        )

        out = {}

        for branch, mp in preds.items():  # branch in {"image", "action"}
            x_start = mp.pred_x_start
            if clip_denoised:
                x_start = x_start.clamp(-1., 1.)

            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_start,
                x_t=noisy_imgs if branch == "image" else noisy_actions,
                t=t,
            )
            out[branch] = dict(
                mean=model_mean,
                var=posterior_variance,
                logvar=posterior_log_variance,
                x_start=x_start,
            )

        return out


    @torch.no_grad()
    def p_sample(
        self,
        noisy_imgs,
        noisy_actions,
        t: int,
        image_primary,
        image_wrist,
        state,
        text_token,
        guidance_weight=0,
    ):
        b, device = noisy_imgs.shape[0], noisy_imgs.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)

        out = self.p_mean_variance(
            noisy_imgs,
            noisy_actions,
            batched_times,
            image_primary,
            image_wrist,
            state,
            text_token,
            clip_denoised=True,
            guidance_weight=guidance_weight,
        )

        ret = {}
        for branch, vals in out.items():
            noise_source = noisy_imgs if branch == "image" else noisy_actions
            noise = torch.randn_like(noise_source) if t > 0 else 0
            pred = vals["mean"] + (0.5 * vals["logvar"]).exp() * noise
            ret[branch] = (pred, vals["x_start"])

        return ret


    @torch.no_grad()
    def p_sample_loop(
        self,
        noisy_imgs_shape,
        noisy_actions_shape,
        image_primary,
        image_wrist,
        state,
        text_token,
        return_all_timesteps=False,
        guidance_weight=0,
    ):
        device = self.betas.device
        b = noisy_imgs_shape[0]

        # initialize noisy inputs
        noisy_imgs = torch.randn(noisy_imgs_shape, device=device)
        noisy_actions = torch.randn(noisy_actions_shape, device=device)

        imgs, actions = [noisy_imgs], [noisy_actions]
        img_x_start, act_x_start = None, None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            out = self.p_sample(
                noisy_imgs,
                noisy_actions,
                t,
                image_primary,
                image_wrist,
                state,
                text_token,
                guidance_weight=guidance_weight,
            )

            noisy_imgs, img_x_start = out["image"]
            noisy_actions, act_x_start = out["action"]

            imgs.append(noisy_imgs)
            actions.append(noisy_actions)

        ret_imgs = noisy_imgs if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret_actions = noisy_actions if not return_all_timesteps else torch.stack(actions, dim=1)

        return self.unnormalize(ret_imgs), ret_actions


    # @torch.no_grad()
    # def ddim_sample(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    #     times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     img = torch.randn(shape, device=device)
    #     imgs = [img]

    #     x_start = None

    #     for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
    #         time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
    #         # self_cond = x_start if self.self_condition else None
    #         pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True, guidance_weight=guidance_weight)

    #         if time_next < 0:
    #             img = x_start
    #             imgs.append(img)
    #             continue

    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]

    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         c = (1 - alpha_next - sigma ** 2).sqrt()

    #         noise = torch.randn_like(img)

    #         img = x_start * alpha_next.sqrt() + \
    #               c * pred_noise + \
    #               sigma * noise

    #         imgs.append(img)

    #     ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    #     ret = self.unnormalize(ret)
    #     return ret

    @torch.no_grad()
    def ddim_sample(
        self,
        noisy_imgs_shape,
        noisy_actions_shape,
        image_primary,
        image_wrist,
        state,
        text_token,
        return_all_timesteps=False,
        guidance_weight=0,
    ):
        batch, device = noisy_imgs_shape[0], self.betas.device
        total_timesteps, sampling_timesteps, eta, objective = (
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        # [-1, 0, 1, 2, ..., T-1]
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # initialize noisy inputs
        noisy_imgs = torch.randn(noisy_imgs_shape, device=device)
        noisy_actions = torch.randn(noisy_actions_shape, device=device)

        imgs, actions = [noisy_imgs], [noisy_actions]
        img_x_start, act_x_start = None, None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            preds = self.model_predictions(
                t=time_cond,
                image_primary=image_primary,
                image_wrist=image_wrist,
                state=state,
                text_token=text_token,
                noisy_imgs=noisy_imgs,
                noisy_actions=noisy_actions,
                clip_x_start=False,
                rederive_pred_noise=True,
                guidance_weight=guidance_weight,
            )

            # update both branches
            updated = {}
            for branch, mp in preds.items():
                if time_next < 0:
                    updated[branch] = mp.pred_x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma**2).sqrt()

                noise_source = noisy_imgs if branch == "image" else noisy_actions
                noise = torch.randn_like(noise_source)

                updated_val = (
                    mp.pred_x_start * alpha_next.sqrt()
                    + c * mp.pred_noise
                    + sigma * noise
                )
                updated[branch] = updated_val

            noisy_imgs, noisy_actions = updated["image"], updated["action"]
            imgs.append(noisy_imgs)
            actions.append(noisy_actions)

        ret_imgs = noisy_imgs if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret_actions = noisy_actions if not return_all_timesteps else torch.stack(actions, dim=1)

        # normalize images, actions left raw
        return self.unnormalize(ret_imgs), ret_actions


    @torch.no_grad()
    def sample(
        self,
        image_primary,
        image_wrist,
        state,
        text_token,
        batch_size=16,
        return_all_timesteps=False,
        guidance_weight=0,
    ):
        # image shape: (B, 3, 2, H, W)  -> 2 views (primary, wrist)
        image_size, channels = self.image_size, self.channels
        noisy_imgs_shape = (batch_size, channels, 2, image_size[0], image_size[1])

        # action shape: (B, action_dim, horizon)
        action_dim = 7  # 6 arm + 1 gripper
        horizon = self.model.action_pred_steps
        noisy_actions_shape = (batch_size, action_dim, horizon)

        # choose sampling function
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        # run sampler
        sampled_imgs, sampled_actions = sample_fn(
            noisy_imgs_shape=noisy_imgs_shape,
            noisy_actions_shape=noisy_actions_shape,
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            return_all_timesteps=return_all_timesteps,
            guidance_weight=guidance_weight,
        )

        return sampled_imgs, sampled_actions


    @torch.no_grad()
    def interpolate(
        self,
        x1_imgs,
        x2_imgs,
        x1_actions,
        x2_actions,
        image_primary,
        image_wrist,
        state,
        text_token,
        t=None,
        lam=0.5,
    ):
        """
        Interpolates between two trajectories (image + action streams),
        keeping the same conditioning as in sampling.
        """
        b, device = x1_imgs.shape[0], x1_imgs.device
        t = default(t, self.num_timesteps - 1)

        assert x1_imgs.shape == x2_imgs.shape
        assert x1_actions.shape == x2_actions.shape

        t_batched = torch.full((b,), t, device=device)

        # noisy versions
        xt1_imgs, xt2_imgs = map(lambda x: self.q_sample(x, t=t_batched), (x1_imgs, x2_imgs))
        xt1_actions, xt2_actions = map(lambda x: self.q_sample(x, t=t_batched), (x1_actions, x2_actions))

        # convex combination in noisy space
        imgs = (1 - lam) * xt1_imgs + lam * xt2_imgs
        actions = (1 - lam) * xt1_actions + lam * xt2_actions

        # denoise jointly with conditioning
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            out = self.p_sample(
                noisy_imgs=imgs,
                noisy_actions=actions,
                t=i,
                image_primary=image_primary,
                image_wrist=image_wrist,
                state=state,
                text_token=text_token,
            )
            imgs, _ = out["image"]
            actions, _ = out["action"]

        return self.unnormalize(imgs), actions



    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, img_start, act_start, t, image_primary, image_wrist, state, text_token, noise_img=None, noise_act=None):
        b = img_start.shape[0]

        # add noise
        noise_img = default(noise_img, lambda: torch.randn_like(img_start))
        noise_act = default(noise_act, lambda: torch.randn_like(act_start))
        
        # print("GGD:  p_losses: noisy_img: ", noise_img.shape)
        # print("GGD:  p_losses: noise_act: ", noise_act.shape)
        x_img = self.q_sample(img_start, t, noise=noise_img)
        x_act = self.q_sample(act_start, t, noise=noise_act)

        # print("GGD:  p_losses: x_img: ", x_img.shape)
        # print("GGD:  p_losses: x_act: ", x_act.shape) 
        # forward through model
        arm_pred, grip_pred, img_pred, _, _, _ = self.model(
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            noisy_imgs=x_img,
            noisy_actions=x_act,
            t=t
        )

        # targets
        if self.objective == 'pred_noise':
            target_img, target_act = noise_img, noise_act
        elif self.objective == 'pred_x0':
            target_img, target_act = img_start, act_start
        elif self.objective == 'pred_v':
            v_img = self.predict_v(img_start, t, noise_img)
            v_act = self.predict_v(act_start, t, noise_act)
            target_img, target_act = v_img, v_act
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # compute losses
        loss_img = self.loss_fn(img_pred, target_img, reduction="mean")
        loss_act = self.loss_fn(torch.cat([arm_pred, grip_pred], dim=-1), target_act, reduction="mean")

        return  0.1 * loss_img + self.args.loss_arm_action_ratio *loss_act, loss_img, loss_act


    def forward(self, img_start, act_start, image_primary, image_wrist, state, text_token):
        b, device = img_start.shape[0], img_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img_start = self.normalize(img_start)

        return self.p_losses(img_start, act_start, t, image_primary, image_wrist, state, text_token)

# trainer class

class TrainerFlow(object):
    def __init__(
        self,
        diffusion_model,
        args,
        opt, 
        lr_scheduler,
        train_set,
        valid_set,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        ema_update_every = 10,
        ema_decay = 0.995,
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        cond_drop_chance = 0.1,
    ):
        super().__init__()

        self.args = args
        self.model = diffusion_model
        self.opt = opt
        self.lr_scheduler = lr_scheduler

        # EMA only on rank 0
        if args.rank == 0:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(args.local_rank)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.cond_drop_chance = cond_drop_chance

        # datasets
        valid_ind = [i for i in range(len(valid_set))][:num_samples]
        self.ds = train_set
        self.valid_ds = Subset(valid_set, valid_ind)

        self.dl = self.ds   # already a DataLoader
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = valid_batch_size,
            shuffle = False,
            pin_memory = True,
            num_workers = 4
        )

        self.step = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.precision in ["fp16", "bf16"]))

    @property
    def device(self):
        return self.args.rank % torch.cuda.device_count()

    def save(self, milestone: int):
        if self.args.rank != 0:
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        data = {
            "step": self.step,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "ema_state_dict": self.ema.state_dict() if hasattr(self, "ema") else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None, 
            "epoch": milestone,
        }

        save_path = self.results_folder / f"model-{milestone}.pt"
        torch.save(data, str(save_path))
        print(f"[Rank 0] Saved checkpoint to {save_path}")

    def load(self, milestone: int):
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.args.local_rank}
        ckpt_path = self.results_folder / f"model-{milestone}.pt"
        data = torch.load(str(ckpt_path), map_location=map_location)

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model.load_state_dict(data["model_state_dict"], strict=False)

        self.opt.load_state_dict(data["opt_state_dict"])
        if self.lr_scheduler is not None and data["lr_scheduler_state_dict"] is not None:
            self.lr_scheduler.load_state_dict(data["lr_scheduler_state_dict"])
        if hasattr(self, "ema") and data["ema_state_dict"] is not None:
            self.ema.load_state_dict(data["ema_state_dict"])
        if "scaler" in data and data["scaler"] is not None and self.scaler is not None:
            self.scaler.load_state_dict(data["scaler"])   # 🔥 restore AMP state

        self.step = data["step"]
        print(f"[Rank {self.args.rank}] Loaded checkpoint from {ckpt_path}")

        return data.get("epoch", 0)

    def sample(
        self,
        image_primary,
        image_wrist,
        state,
        text_token,
        batch_size=1,
        guidance_weight=0,):

        device = self.device
        return self.ema.ema_model.sample(
                image_primary=image_primary.to(device),
                image_wrist=image_wrist.to(device),
                state=state.to(device),
                text_token=text_token.to(device),
                batch_size=batch_size,
                guidance_weight=guidance_weight,
            )


    # def train(self):
    #     accelerator = self.accelerator
    #     device = accelerator.device

    #     with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

    #         while self.step < self.train_num_steps:

    #             total_loss = 0.

    #             for _ in range(self.gradient_accumulate_every):
    #                 x, x_cond, goal = next(self.dl)
    #                 x, x_cond = x.to(device), x_cond.to(device)

    #                 goal_embed = self.encode_batch_text(goal)
    #                 ### zero whole goal_embed if p < self.cond_drop_chance
    #                 goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()


    #                 with self.accelerator.autocast():
    #                     loss = self.model(x, x_cond, goal_embed)
    #                     loss = loss / self.gradient_accumulate_every
    #                     total_loss += loss.item()

    #                     self.accelerator.backward(loss)

    #             accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

    #             scale = self.accelerator.self.scaler.get_scale()
                
    #             pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')

    #             accelerator.wait_for_everyone()

    #             self.opt.step()
    #             self.opt.zero_grad()

    #             accelerator.wait_for_everyone()

    #             self.step += 1
    #             if accelerator.is_main_process:
    #                 self.ema.update()

    #                 if self.step != 0 and self.step % self.save_and_sample_every == 0:
    #                     self.ema.ema_model.eval()

    #                     with torch.no_grad():
    #                         milestone = self.step // self.save_and_sample_every
    #                         batches = num_to_groups(self.num_samples, self.valid_batch_size)
    #                         ### get val_imgs from self.valid_dl
    #                         x_conds = []
    #                         xs = []
    #                         task_embeds = []
    #                         for i, (x, x_cond, label) in enumerate(self.valid_dl):
    #                             xs.append(x)
    #                             x_conds.append(x_cond.to(device))
    #                             task_embeds.append(self.encode_batch_text(label))
                            
    #                         with self.accelerator.autocast():
    #                             all_xs_list = list(map(lambda n, c, e: self.ema.ema_model.sample(batch_size=n, x_cond=c, task_embed=e), batches, x_conds, task_embeds))
                        
    #                     print_gpu_utilization()
                        
    #                     gt_xs = torch.cat(xs, dim = 0) # [batch_size, 3*n, 120, 160]
    #                     # make it [batchsize*n, 3, 120, 160]
    #                     n_rows = gt_xs.shape[1] // 3
    #                     gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)
    #                     ### save images
    #                     x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
    #                     # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
    #                     all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
    #                     all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)

    #                     gt_first = gt_xs[:, :1]
    #                     gt_last = gt_xs[:, -1:]



    #                     if self.step == self.save_and_sample_every:
    #                         os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
    #                         gt_img = torch.cat([gt_first, gt_last, gt_xs], dim=1)
    #                         gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
    #                         utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=n_rows+2)

    #                     os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
    #                     pred_img = torch.cat([gt_first, gt_last,  all_xs], dim=1)
    #                     pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
    #                     utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows+2)

    #                     self.save(milestone)

    #             pbar.update(1)

    #     accelerator.print('training complete')
    def train_one_epoch(self, epoch, wandb_logger, args):
        device = self.device
        num_batches_per_epoch_calvin = self.dl.num_batches
        num_batches_per_epoch = num_batches_per_epoch_calvin
        total_training_steps = num_batches_per_epoch * args.num_epochs
        autocast = get_autocast(args.precision)
        cast_dtype = get_cast_dtype(args.precision)

        self.model.train()
        total_loss = 0.0

        # logging meters
        step_time_m, data_time_m = AverageMeter(), AverageMeter()
        end = time.time()
        mv_avg_loss = []
        

        # loop through one epoch only
        pbar = tqdm(
            enumerate(self.dl),
            disable=args.rank != 0,
            total=total_training_steps,
            initial=(epoch * num_batches_per_epoch),
        )
        pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
        mv_avg_loss = []

        for num_steps, batch_calvin in pbar:
            data_time_m.update(time.time() - end)
            global_step = num_steps + epoch * num_batches_per_epoch

            images_primary = batch_calvin[0].to(device, dtype=cast_dtype, non_blocking=True)
            images_wrist = batch_calvin[3].to(device, dtype=cast_dtype, non_blocking=True)
            # text tokens
            text_tokens = batch_calvin[1].to(device, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
            
            # states
            states = batch_calvin[4].to(device, dtype=cast_dtype, non_blocking=True)
            if args.gripper_width:
                input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
            else:
                input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
                input_states[..., 6:] = (input_states[..., 6:] + 1) // 2
            
            # self_key_point
            self_keypoints = None
            
            # actions
            actions = batch_calvin[2].to(device, dtype=cast_dtype, non_blocking=True)
            # label. [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
            actions[..., 6:] = (actions[..., 6:] + 1) // 2
            input_image_primary = images_primary[:, :args.sequence_length, :]
            input_image_wrist = images_wrist[:, :args.sequence_length, :]
            input_text_token = text_tokens[:, :args.sequence_length, :]
            input_state = input_states[:, :args.sequence_length, :]

            # label action
            label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 
            act_start = label_actions.permute(0, 3, 2, 1).contiguous() #(B, action_dim, horizon, seq_len)
            # print("Shape of act start:",act_start.shape)
            label_image_primary = images_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal]
            label_image_wrist   = images_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal]
            img_start = torch.stack([label_image_primary, label_image_wrist], dim=2)
            # forward pass
            with autocast():
                loss, loss_img, loss_act = self.model(
                    img_start=img_start,      # or rearrange if needed
                    act_start=act_start,
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                )

            loss = loss / args.gradient_accumulation_steps
            loss_img = loss_img / args.gradient_accumulation_steps
            loss_act = loss_act / args.gradient_accumulation_steps
            total_loss += loss.item()
            mv_avg_loss.append(loss.item())

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            # optimizer step
            if ((num_steps + 1) % args.gradient_accumulation_steps) == 0 or (
                num_steps == num_batches_per_epoch - 1
                ):
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                self.lr_scheduler.step()
                
                if args.rank and hasattr(self, "ema"):
                    self.ema.update()

            # logging
            step_time_m.update(time.time() - end)
            end = time.time()
            avg_horizon = min(100, len(mv_avg_loss))

            pbar.set_postfix(
                {
                    "avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon,
                    "loss": loss.item(),
                }
            )

            if wandb_logger and args.rank == 0:
                wandb_logger.log(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": loss.item() * args.gradient_accumulation_steps,
                        "loss_image": loss_img.item() * args.gradient_accumulation_steps,
                        "avg_loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon,
                        "loss_arm_action": loss_act.item() * args.gradient_accumulation_steps,
                        "lr": self.opt.param_groups[0]["lr"],
                        "loss_scale": self.scaler.get_scale(),   # 🔥 log this
                    }
                )

        # ===== Validation + Sampling =====
        # if accelerator.is_main_process:
        #     self.ema.ema_model.eval()
            # with torch.no_grad():
            #     for i, (val_imgs_primary, val_text, val_actions, val_imgs_wrist, val_states) in enumerate(self.valid_dl):
            #         val_imgs_primary = val_imgs_primary.to(device)
            #         val_imgs_wrist = val_imgs_wrist.to(device)
            #         val_states = val_states.to(device)
            #         val_text = val_text.to(device)

            #         sampled_imgs, sampled_actions = self.ema.ema_model.sample(
            #             image_primary=val_imgs_primary,
            #             image_wrist=val_imgs_wrist,
            #             state=val_states,
            #             text_token=val_text,
            #             batch_size=val_imgs_primary.size(0),
            #         )

            #         # save predicted images
            #         os.makedirs(str(self.results_folder / "imgs/outputs"), exist_ok=True)
            #         utils.save_image(
            #             sampled_imgs,
            #             str(self.results_folder / f"imgs/outputs/sample-epoch{epoch+1}.png"),
            #             nrow=4,
            #         )
            #         break  # only sample one batch

            # checkpoint
        
        self.save(milestone=epoch+1)

        print(f"Epoch {epoch+1} complete, avg loss {total_loss/len(self.ds):.4f}")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count