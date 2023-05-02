import torch
from model import Generator, Discriminator
from torchvision import utils
import argparse, math, random, os, torch
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm
from op import conv2d_gradfix
from model import Generator, Discriminator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from clip import CLIPText
from datetime import datetime
import wandb
import lpips
# style_dim = 256
# tin_dim = 768
# tout_dim = 256
# image_size = 64
# batch = 1
# n_mlp = 8
# seq_len = 18

# with torch.no_grad():
#     g = Generator(image_size, style_dim, n_mlp, tin_dim, tout_dim)
#     z = torch.randn(batch, style_dim)
#     text_embeds = torch.randn(batch, seq_len, tin_dim)
#     images = g(z, text_embeds)[0]
#     for i in range(len(images)):
#         print(images[i].shape)
#         utils.save_image(images[i], f"test-{i}.png", nrow=1, normalize=True, value_range=(-1, 1))

#     d = Discriminator(image_size, tin_dim, tout_dim)
#     out = d(images, text_embeds)
def sample_img(args, text_encoder, generator, text_prompt):
    sample_z = torch.randn(1 , args.latent, device=device)
    text_embeds = text_encoder(text_prompt)
    
    sample, _ = generator([sample_z], text_embeds)
    # wandb_save(sample[-1],'Evaluation', i)
    utils.save_image(sample[-1], f"sample/{text_prompt}.png",)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GigaGAN trainer")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path", type=str, default="lambdalabs/pokemon-blip-captions")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=64, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0025, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"{current_time}",
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()
    # args.dataset_name = "dream-textures/textures-color-1k"
    args.dataset_name = "lambdalabs/pokemon-blip-captions"
    args.latent = 128
    args.n_mlp = 8
    args.start_iter = 0
    args.tin_dim = 512
    args.tout_dim = 1024
    args.use_multi_scale = False
    args.use_text_cond = True
    args.use_self_attn = True
    # args.sample_s
    args.n_sample = 1
    args.batch = 2
    args.save_every = 10000
    args.sample_every = 200
    args.use_noise = True
    args.use_matching_loss = True
    args.run_name = f"use_matching_loss {args.use_matching_loss} use_text_cond ({args.use_text_cond}) use_self_attn ({args.use_self_attn}) use_noise ({args.use_noise} current_time {current_time})"
    device = args.device
    args.ckpt = "checkpoint/use_matching_loss True use_text_cond (True) use_self_attn (True) use_noise (True current_time 0429_205603)_best.pt"
    args.ckpt = "checkpoint/use_matching_loss True use_text_cond (True) use_self_attn (True) use_noise (True current_time 0429_172632)_best.pt"
    args.ckpt = "checkpoint/400000.pt"
    generator = Generator(
        args.size, args.latent, args.n_mlp, args.tin_dim, args.tout_dim,
        channel_multiplier=args.channel_multiplier, use_multi_scale=args.use_multi_scale,
        use_text_cond=args.use_text_cond, use_noise= args.use_noise
    ).to(device)
    
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        # discriminator.load_state_dict(ckpt["d"])
        # g_ema.load_state_dict(ckpt["g_ema"])

        # g_optim.load_state_dict(ckpt["g_optim"])
        # d_optim.load_state_dict(ckpt["d_optim"])

    generator.eval()
    text_encoder = CLIPText(args.device)
    
    z = torch.randn(args.n_sample, args.latent, device=device)
    text_name = input("Enter text: ")
    text_prompt = [text_name]
    sample_img(args, text_encoder, generator, text_prompt)
