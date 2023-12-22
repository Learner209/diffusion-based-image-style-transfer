import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from sdxl.models.diffusion.ddim import DDIMSampler
from sdxl.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from sdxl.util import exists, instantiate_from_config

from torch.cuda import amp
import torch.nn as nn
import argparse

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

torch.set_grad_enabled(False)

DATA_PARALLEL = False
GPUS = [2, 3]
if not DATA_PARALLEL:
    DEVICE = "cuda:3"


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    if DATA_PARALLEL:
        model = nn.DataParallel(model, device_ids=GPUS)
        model = model.module
    else:
        model = model.to(device)

    sampler = DDIMSampler(model)
    return sampler


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        "lr": rearrange(image, 'h w c -> 1 c h w'),
        "txt": num_samples * [txt],
    }
    batch["lr"] = repeat(batch["lr"].to(device=device),
                         "1 ... -> n ...", n=num_samples)
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    x_low = x_low.to(memory_format=torch.contiguous_format).float()

    device = torch.device(
        DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    if DATA_PARALLEL:
        model.low_scale_model = nn.DataParallel(model.low_scale_model, device_ids=GPUS)
        model.low_scale_model = model.low_scale_model.module
    else:
        model.low_scale_model = model.low_scale_model.to(device)

    x_low = x_low.to(device)
    noise_level = noise_level.to(device)
    print("In the make_noise_augmentation function")
    print(x_low.device, noise_level.device)
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


def paint(sampler, image, prompt, seed, scale, h, w, steps, num_samples=1, callback=None, eta=0., noise_level=None):
    device = torch.device(
        DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    model = sampler.model

    if DATA_PARALLEL:
        model = nn.DataParallel(model, device_ids=GPUS)
        model = model.module
        model.cond_stage_model = nn.DataParallel(model.cond_stage_model, device_ids=GPUS)
        model.cond_stage_model = model.cond_stage_model.module
        model.first_stage_model = nn.DataParallel(model.first_stage_model, device_ids=GPUS)
        model.first_stage_model = model.first_stage_model.module
    else:
        model = model.to(device)
        model.cond_stage_model = model.cond_stage_model.to(device)
        model.first_stage_model = model.first_stage_model.to(device)

    seed_everything(seed)
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, model.channels, h, w)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), amp.autocast(enabled=True):
        batch = make_batch_sd(
            image, txt=prompt, device=device, num_samples=num_samples)
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                cc = cc.to(device)
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                   p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_augment, noise_level = make_noise_augmentation(
                model, batch, noise_level)
            cond = {"c_concat": [x_augment],
                    "c_crossattn": [c], "c_adm": noise_level}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [x_augment], "c_crossattn": [
                uc_cross], "c_adm": noise_level}
        else:
            raise NotImplementedError()

        print("In the paint function")
        uc_full["c_crossattn"][0] = uc_full["c_crossattn"][0].to(device)
        print(cond["c_adm"].device, cond["c_concat"][0].device, cond["c_crossattn"][0].device, start_code.device, uc_full["c_adm"].device, uc_full["c_concat"][0].device, uc_full["c_crossattn"][0].device)
        shape = [model.channels, h, w]
        samples, intermediates = sampler.sample(
            steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
            callback=callback
        )
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [Image.fromarray(img.astype(np.uint8)) for img in result]


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def predict(sampler, input_image_path, prompt, steps, num_samples, scale, seed, eta, noise_level):
    device = torch.device(
        DEVICE) if torch.cuda.is_available() else torch.device("cpu")

    init_image = Image.open(input_image_path).convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32
    width, height = image.size

    noise_level = torch.Tensor(
        num_samples * [noise_level]).to(device).long()
    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    result = paint(
        sampler=sampler,
        image=image,
        prompt=prompt,
        seed=seed,
        scale=scale,
        h=height, w=width, steps=steps,
        num_samples=num_samples,
        callback=None,
        noise_level=noise_level
    )
    return result


if __name__ == "__main__":
    import glob
    import os
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description="Image Super-Resolution Inference")
    parser.add_argument("--config", default="/mnt/homes/minghao/AI/final_project/sdxl/configs/stable-diffusion/x4-upscaling.yaml", help="Path to model config file")
    parser.add_argument("--checkpoint", default="/mnt/homes/minghao/AI/final_project/sdxl/pretrained_weights/x4-upscaler-ema.ckpt", help="Path to model checkpoint")
    parser.add_argument("--image", default="/mnt/homes/minghao/AI/final_project/datasets/photo_mural_data/photo/0a271892d0.jpg", help="")
    parser.add_argument("--prompt", default="mural style", help="Prompt for image generation")
    parser.add_argument("--steps", type=int, default=150, help="DDIM Steps")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of Samples")
    parser.add_argument("--scale", type=float, default=15.0, help="Scale")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--eta", type=float, default=0.0, help="eta (DDIM)")
    parser.add_argument("--noise_level", type=int, default=20, help="Noise Augmentation")
    out_folder = "/mnt/homes/minghao/AI/final_project/src/im_variations/super_resolution"
    args = parser.parse_args()

    sampler = initialize_model(args.config, args.checkpoint)
    result_images = predict(
        sampler, args.image, args.prompt, args.steps, args.num_samples, args.scale, args.seed, args.eta, args.noise_level
    )
    output_path = "/mnt/homes/minghao/AI/final_project/src/im_variations/super_resolution"
    root_path = "/mnt/homes/minghao/AI/final_project/src/im_variations/samples"

    image_paths = glob.glob(os.path.join(root_path, "*.png"))
    for image_path in tqdm(image_paths):
        assert os.path.exists(image_path)
        result_images = predict(
            sampler, image_path, args.prompt, args.steps, args.num_samples, args.scale, args.seed, args.eta, args.noise_level
        )
        base_name = os.path.basename(image_path)
        base_name = os.path.splitext(base_name)[0]
        if os.path.exists(os.path.join(output_path, f"{base_name}_03.png")):
            continue

        for idx, img in enumerate(result_images):
            print("Saving image to: ", os.path.join(output_path, f"{base_name}_03.png"))
            img.save(os.path.join(output_path, f"{base_name}_{idx:02}.png"))
