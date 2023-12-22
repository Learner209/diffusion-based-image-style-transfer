from io import BytesIO
import os
from contextlib import nullcontext
import glob

import fire
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
import requests
import pandas as pd

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from tqdm import tqdm
import torch.nn as nn
from torch.cuda import amp

import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

torch.set_grad_enabled(False)
DATA_PARALLEL = False
if DATA_PARALLEL:
    DEVICE_IDS = [2, 3]


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        # print("missing keys:")
        # print(m)
        pass
    if len(u) > 0 and verbose:
        # print("unexpected keys:")
        # print(u)
        pass

    if DATA_PARALLEL:
        model = nn.DataParallel(model, device_ids=DEVICE_IDS)
        model = model.module
        # model.cond_stage_model = nn.DataParallel(model.cond_stage_model, device_ids=DEVICE_IDS)
        # model.cond_stage_model = model.cond_stage_model.module
        # model.first_stage_model = nn.DataParallel(model.first_stage_model, device_ids=DEVICE_IDS)
        # model.first_stage_model = model.first_stage_model.module
        # model.cond_stage_model.to(device)
        # model.to(device)

    else:
        model.to(device)

    model.eval()
    return model


def load_im(im_path):
    if im_path.startswith("http"):
        response = requests.get(im_path)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(im_path).convert("RGB")
    tforms = transforms.Compose([
        # transforms.Resize(224),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    inp = tforms(im).unsqueeze(0)
    return inp * 2 - 1


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta):
    precision_scope = autocast if precision == "autocast" else nullcontext
    with autocast("cuda"): # amp.autocast(True):
        with model.ema_scope():
            print("Getting learned conditioning....")
            print("model's device:", model.device)

            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)

            if scale != 1.0:
                uc = torch.zeros_like(c)
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=True,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)


def main(
    im_path="/mnt/homes/minghao/AI/final_project/datasets/photo_mural_data/photo/*.jpg",
    ckpt="/mnt/homes/minghao/AI/final_project/src/pretrained_weights/sd-clip-vit-l14-img-embed_ema_only.ckpt",
    config="/mnt/homes/minghao/AI/final_project/src/configs/stable-diffusion/sd-image-condition-finetune.yaml",
    outpath="/mnt/homes/minghao/AI/final_project/src/im_variations",
    scale=3.0,
    h=512,
    w=512,
    n_samples=10,
    precision="fp32",
    plms=False,
    ddim_steps=120,
    ddim_eta=0.0,
    device_idx=2,
    save=True,
    eval=True,
):

    device = f"cuda:{device_idx}"
    print("In the main function, device is", device)
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)
    print("Model loaded: ", model.device)

    if plms:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model)

    if DATA_PARALLEL:
        sampler = nn.DataParallel(sampler, device_ids=DEVICE_IDS)
        sampler = sampler.module
    else:
        sampler.to(device)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    if isinstance(im_path, str):
        im_paths = glob.glob(im_path)

    im_paths = sorted(im_paths)
    print(f"Found {len(im_paths)} images")
    all_similarities = []
    for im in tqdm(im_paths, desc="Generating samples"):

        if DATA_PARALLEL:
            input_im = load_im(im)
        else:
            input_im = load_im(im).to(device)

        print("Inferenceing on image: device", input_im.device)

        basename = os.path.basename(im)
        basename = os.path.splitext(basename)[0]
        filename = os.path.join(sample_path, f"{basename}_00009.png")
        if os.path.exists(filename):
            continue

        x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta)
        cnt = 0

        for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            filename = os.path.join(sample_path, f"{basename}_{cnt:05}.png")
            Image.fromarray(x_sample.astype(np.uint8)).save(filename)
            cnt += 1

        if eval:
            generated_embed = model.get_learned_conditioning(x_samples_ddim).squeeze(1)
            prompt_embed = model.get_learned_conditioning(input_im).squeeze(1)

            generated_embed /= generated_embed.norm(dim=-1, keepdim=True)
            prompt_embed /= prompt_embed.norm(dim=-1, keepdim=True)
            similarity = prompt_embed @ generated_embed.T
            mean_sim = similarity.mean()
            all_similarities.append(mean_sim.unsqueeze(0))

    df = pd.DataFrame(zip(im_paths, [x.item() for x in all_similarities]), columns=["filename", "similarity"])
    df.to_csv(os.path.join(sample_path, "eval.csv"))
    print(torch.cat(all_similarities).mean())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--finetuned_ckpt_path",
        type=str,
        default="/mnt/homes/minghao/AI/final_project/src/pretrained_weights/sd-clip-vit-l14-img-embed_ema_only.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/homes/minghao/AI/final_project/src/im_variations"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="/mnt/homes/minghao/AI/final_project/src/configs/stable-diffusion/sd-image-condition-finetune.yaml",
        help="path to the input image"
    )
    parser.add_argument(
        "--gpu",
        default='0',
    )
    # region
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml",
        help="path to config which constructs model",
    )
    # endregion

    opt = parser.parse_args()

    assert os.path.exists(opt.output_path) and os.path.exists(opt.config_path) and os.path.exists(opt.finetuned_ckpt_path)
    assert int(opt.gpu) in [2, 3]
    assert opt.output_path != "/mnt/homes/minghao/AI/final_project/src/im_variations"

    main(im_path="/mnt/homes/minghao/AI/final_project/datasets/photo_mural_data/test/*.jpg",
         ckpt=opt.finetuned_ckpt_path,
         config=opt.config_path,
         outpath=opt.output_path,
         scale=3.0,
         h=512,
         w=512,
         n_samples=10,
         precision="fp16",
         plms=False,
         ddim_steps=120,
         ddim_eta=0.0,
         device_idx=opt.gpu,
         save=True,
         eval=True)
