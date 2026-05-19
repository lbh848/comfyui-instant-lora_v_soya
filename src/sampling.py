from __future__ import annotations

import torch
import comfy.model_management
import comfy.sd
import comfy.sample
import comfy.utils
import comfy.lora


def _classify_lora_key(k: str) -> str:
    """LoRA 키를 U-Net 블록 그룹으로 분류.
    Returns: "IN0"-"IN8", "MID0"-"MID2", "OUT0"-"OUT8" 또는 "BASE"
    """
    prefix = "diffusion_model."
    if not k.startswith(prefix):
        return "BASE"
    k_unet = k[len(prefix):]
    for block_type in ("input_blocks", "middle_block", "output_blocks"):
        if k_unet.startswith(block_type + "."):
            num_str = k_unet[len(block_type) + 1:]
            dot_pos = num_str.find(".")
            num_part = num_str[:dot_pos] if dot_pos >= 0 else num_str
            if num_part.isdigit():
                if block_type == "input_blocks":
                    return f"IN{num_part}"
                elif block_type == "middle_block":
                    return f"MID{num_part}"
                else:
                    return f"OUT{num_part}"
    return "BASE"


def patch_lora_onto_models(
    model, clip, lora_path: str,
    strength_model: float, strength_clip: float,
    block_weight: str = "",
):
    """동적으로 LoRA를 패치하고 (patched_model, patched_clip) 튜플을 반환한다."""
    if not block_weight:
        # 기존 동작: uniform strength
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        m, c = comfy.sd.load_lora_for_models(model, clip, lora_sd, strength_model, strength_clip)
        del lora_sd
        return m, c

    # Per-block weight: LBW 방식
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model)
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    del lora_sd

    # 블록 가중치 파싱
    bw_values = [float(v.strip()) for v in block_weight.split(",") if v.strip()]

    # 블록 순서 정의: BASE, IN0..IN11, MID0..MID3, OUT0..OUT11
    ordered_groups = ["BASE"]
    for i in range(12):
        ordered_groups.append(f"IN{i}")
    for i in range(4):
        ordered_groups.append(f"MID{i}")
    for i in range(12):
        ordered_groups.append(f"OUT{i}")

    # 각 그룹에 가중치 매핑
    group_ratio: dict[str, float] = {}
    for idx, g in enumerate(ordered_groups):
        if idx < len(bw_values):
            group_ratio[g] = bw_values[idx]
        else:
            group_ratio[g] = bw_values[-1] if bw_values else 1.0

    # Per-key patching
    new_model = model.clone()
    new_clip = clip.clone()

    for key, val in loaded.items():
        k = key if isinstance(key, str) else key[0]
        group = _classify_lora_key(k)
        ratio = group_ratio.get(group, 1.0)
        if ratio == 0:
            continue  # muted

        is_clip = "text" in k or "encoder" in k
        if is_clip:
            new_clip.add_patches({key: val}, strength_clip * ratio)
        else:
            new_model.add_patches({key: val}, strength_model * ratio)

    return new_model, new_clip


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
