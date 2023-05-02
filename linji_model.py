import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from layers import (
    PixelNorm, make_kernel, Upsample, Downsample, Blur, EqualConv2d,
    ModulatedConv2d, EqualLinear, NoiseInjection,
    SelfAttention, CrossAttention, TextEncoder,
)

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, style_dim,
        n_kernel=1, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel, out_channel, kernel_size, style_dim, n_kernel=n_kernel,
            upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out
        
class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out
    
class Generator(nn.Module):
    def __init__(self, size, constant_dim, z_dim, t_dim, n_mlp, use_text_cond = True, 
                 use_self_attn = False, use_cross_attn = False, use_multi_scale = False,
                 lr_mlp=0.01, blur_kernel=[1, 3, 3, 1], 
                 ) -> None:
        super().__init__()
        
        self.size = size
        self.constant_dim = constant_dim
        self.z_dim = z_dim
        self.t_dim = t_dim
        self.n_mlp = n_mlp
        self.use_text_cond = use_text_cond
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn
        self.use_multi_scale = use_multi_scale
        
        if self.use_text_cond:
            self.text_encoder = TextEncoder(512, self.t_dim)
            self.style_dim = self.z_dim + self.t_dim
        else:
            self.style_dim = self.z_dim
        
        layers = [PixelNorm()]
        
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    self.style_dim, self.style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
        # out of layers is w = self.style_dim = self.z_dim + self.t_dim
        
        self.input = ConstantInput(self.constant_dim)
        
        self.conv1 = StyledConv(
            512, 512, 1, self.style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(512, self.style_dim, upsample=False)
        
        self.num_layers = 9
        
        self.convs = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            # self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            res = 2 ** i
            out_channel = self.channels[res]

            self.convs.append(StyledConv(
                in_channel, out_channel, 3, self.style_dim, upsample=True,
                blur_kernel=blur_kernel, n_kernel=n_kernels[res],
            ))
            self.convs.append(StyledConv(
                out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel,
                n_kernel=n_kernels[res],
            ))

            self.attns.append(
                SelfAttention(out_channel) if use_self_attn and res in attn_res else None
            )
            self.attns.append(
                CrossAttention(out_channel, tout_dim) if use_text_cond and res in attn_res else None
            )

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))

            in_channel = out_channel       