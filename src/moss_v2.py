"""Use the connected Comfy encoders with the installed anima_lora MOSS trainer."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import runpy
import sys
import tempfile

import comfy.model_management
from filelock import FileLock
import torch
from safetensors.torch import save_file

from .runtime import IMAGE_EXTENSIONS, ensure_dir, plugin_root, run_command, write_json


def cached_checkpoint(source: Path, models: Path, variant: str, engine: Path, log_path: Path, env: dict) -> Path:
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    destination = models / digest
    with FileLock(str(models / (digest + ".lock"))):
        if not destination.is_dir():
            # Only the conversion subprocess needs exclusive VRAM. Cached inputs
            # can use Comfy's usual memory management until the trainer starts.
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            with tempfile.TemporaryDirectory(dir=models) as stage:
                candidate = Path(stage) / "model"
                run_command([sys.executable, str(Path(__file__).with_name("moss_v2_checkpoint.py")),
                             str(source), str(candidate), digest, variant], cwd=engine, log_path=log_path, env=env)
                candidate.rename(destination)
        else:
            print(f"[MOSS V2] Reusing converted checkpoint {digest[:16]}", flush=True)
    return destination


@torch.inference_mode()
def cache_dataset(source: Path, cache: Path, clip, vae, resolution: int, engine: Path):
    # Load V2's standalone image sizing functions without importing its top-level
    # 'library' package into the shared Comfy process.
    sizing = runpy.run_path(str(engine / "library/moss/caching/images.py"))
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError("MOSS V2 requires at least one training image.")
    dataset = []
    for i, image in enumerate(files):
        caption = image.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
        encoded = clip.encode_from_tokens(clip.tokenize(caption), return_dict=True)
        embeds = encoded["cond"][0]
        mask = encoded.get("attention_mask")
        if mask is not None:
            embeds = embeds[mask.reshape(-1).bool().to(embeds.device)]
        if embeds.ndim != 2 or embeds.shape[1] != 4096 or not len(embeds):
            raise ValueError("MOSS V2 requires Qwen3.5-9B last hidden state embeddings.")
        save_file({"embeds": embeds.cpu().to(torch.bfloat16).contiguous()},
                  cache / f"{i}.moss_te.safetensors", metadata={"caption": caption, "source": str(image)})
        with image.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        dataset.append({"image": str(image), "sha256": digest, "caption": caption})
    for i, image in enumerate(files):
        with sizing["load_rgb"](image) as rgb:
            width, height = sizing["target_size"](*rgb.size, resolution)
            pixels = sizing["to_pixels"](sizing["fit"](rgb, width, height))
        latent = vae.encode(((pixels + 1) / 2).movedim(1, -1))[0].float().cpu()
        if latent.shape != (128, height // 16, width // 16):
            raise ValueError("MOSS V2 requires the packed, normalized FLUX.2 VAE latents.")
        save_file({"latent": latent.contiguous()}, cache / f"{i}_{width}x{height}.moss_latent.safetensors",
                  metadata={"width": str(width), "height": str(height), "source": str(image)})
    return dataset


def prepare_training(checkpoint, clip, vae, builtins, run_dir, options, settings, log_path):
    from .nodes import _format_toml_value

    root = plugin_root()
    runpy.run_path(str(root / "install.py"), run_name="__main__")
    engine = root / "runtime/anima_lora"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(engine), str(root / "runtime/anime_tools"))),
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "TORCHINDUCTOR_CACHE_DIR": str(ensure_dir(root / "runtime/moss_v2/compile")),
           "TRITON_CACHE_DIR": str(ensure_dir(root / "runtime/moss_v2/triton"))}
    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        raise ValueError("MOSS INT8 V2 requires a CUDA device.")
    # The upstream subprocess uses cuda:0. Keep the worker's GPU assignment.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[device_index].strip() if visible else str(device_index)
    variant = "base" if Path(checkpoint).name.casefold() == "moss_image_base_int8_convrot.safetensors" else "sft"
    model_dir = cached_checkpoint(Path(checkpoint), ensure_dir(root / "runtime/moss_v2/models"),
                                  variant, engine, log_path, env)
    cache = ensure_dir(run_dir / "embeddings")
    dataset = cache_dataset(Path(builtins["TRAIN_DIR"]), cache, clip, vae,
                            int(options.resolution_override or 1024), engine)
    epochs = int(settings.get("max_train_epochs", 0))
    total = epochs * len(dataset) if epochs else options.steps_override
    if total < 1:
        raise ValueError("Select a positive epoch or step budget for MOSS V2.")
    optimizer = options.optimizer_type if options.optimizer_type != "default" else "Muon"
    if optimizer not in {"Muon", "AdamW"}:
        raise ValueError("MOSS V2 supports Muon and AdamW.")
    config = {
        "model": variant, "model_dir": str(model_dir), "int8": "stream", "int8_file": str(model_dir / "forward.safetensors"),
        "cache": str(cache), "output": str(Path(builtins["OUTPUT_DIR"]) / (builtins["OUTPUT_NAME"] + ".safetensors")),
        "epochs": epochs, "max_steps": 0 if epochs else total, "save_every_steps": options.save_every_n_steps,
        "rank": options.network_dim_override or 32, "alpha": options.network_alpha_override or 128,
        "lr": options.learning_rate_override or 1e-4, "optimizer": optimizer.lower(),
        "grad_accum": 1, "down_init": "kaiming", "max_grad_norm": 1.0, "caption_dropout": 0.0,
        "warmup_ratio": settings.get("warmup_ratio", 0.05), "ema_window": settings.get("ema_window", 0.1),
        "stream": settings.get("stream", "fp32"), "lora_dtype": "bf16", "repa_weight": 0.0,
        "seed": options.seed_override if options.seed_override >= 0 else 42,
    }
    config_path = run_dir / "train.toml"
    config_path.write_text("".join(f"{k} = {_format_toml_value(v)}\n" for k, v in config.items()), encoding="utf-8")
    trainer = Path(__file__).with_name("moss_v2_training.py")
    command = [sys.executable, str(trainer), "--config", str(config_path)]
    # Keep the trainer's progress stream alongside the existing per-run log.
    env["ANIMA_DAEMON_JOB_DIR"] = str(run_dir)
    write_json(run_dir / "moss-runtime.json", {
        "source_checkpoint": checkpoint, "source_sha256": model_dir.name, "dataset": dataset,
        "model": {str(p.relative_to(engine)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (engine / "library/moss/model.py", engine / "library/moss/loader.py",
                            engine / "library/moss/dit_int8/base.py")},
        "training_adapter_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
        "preprocessing": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), engine / "library/moss/caching/images.py")},
        "checkpoint_adapter_sha256": hashlib.sha256(Path(__file__).with_name("moss_v2_checkpoint.py").read_bytes()).hexdigest(),
        "cwd": str(engine), "command": command, "config": str(config_path), "seed": config["seed"],
    })
    return config_path, (command, engine, env), total
