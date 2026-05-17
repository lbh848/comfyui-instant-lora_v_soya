from __future__ import annotations

import torch
import comfy.model_management
import comfy.sd
import comfy.sample
import comfy.utils


def patch_lora_onto_models(model, clip, lora_path: str, strength_model: float, strength_clip: float):
    """동적으로 LoRA를 패치하고 (patched_model, patched_clip) 튜플을 반환한다."""
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    return comfy.sd.load_lora_for_models(model, clip, lora_sd, strength_model, strength_clip)


def encode_conditioning(clip, text: str):
    """CLIP을 이용해 텍스트를 Conditioning으로 인코딩."""
    if clip is None:
        raise RuntimeError("CLIP is None for conditioning")
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def generate_empty_latent(width: int, height: int, batch_size: int = 1):
    """빈 잠재 이미지(Latent)를 생성."""
    device = comfy.model_management.intermediate_device()
    latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=device)
    return {"samples": latent, "downscale_ratio_spacial": 8}


def sample_latent(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
    """KSampler를 이용해 Latent를 denoising."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))
    noise = comfy.sample.prepare_noise(latent_image, seed, None)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=denoise, disable_noise=False,
        noise_mask=None, callback=None, disable_pbar=disable_pbar, seed=seed
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out


def decode_latent(vae, latent):
    """VAE를 이용해 Latent를 픽셀 이미지로 디코딩."""
    latent_samples = latent["samples"]
    if latent_samples.is_nested:
        latent_samples = latent_samples.unbind()[0]
    images = vae.decode(latent_samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def generate_preview(
    model, clip, vae,
    positive_text: str, negative_text: str,
    width: int, height: int,
    seed: int, steps: int, cfg: float,
    sampler_name: str, scheduler: str,
):
    """
    전체 preview 파이프라인.
    반환: torch.Tensor [1, H, W, C] (RGB, float32, 0~1)
    """
    positive = encode_conditioning(clip, positive_text)
    negative = encode_conditioning(clip, negative_text)
    latent = generate_empty_latent(width, height, batch_size=1)
    latent = sample_latent(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0)
    images = decode_latent(vae, latent)
    return images
