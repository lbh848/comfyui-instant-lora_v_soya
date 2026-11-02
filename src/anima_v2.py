"""Run the Anima trainer's public CLIs in ComfyUI's Python environment."""

from __future__ import annotations

import json
import math
import os
import sys
import tomllib
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .runtime import IMAGE_EXTENSIONS, ensure_dir, plugin_root, run_command


def prepare_training(profile, slots, builtins, run_dir, options, v2_options, log_path):
    # Reuse the caller's dataset, resolved model paths and output directory.
    # The upstream trainer consumes a dataset blueprint instead of sd-scripts'
    # repeat-folder config, so it needs its own config at this boundary.
    from .nodes import _format_toml_value

    engine = Path(v2_options.get("engine_path") or plugin_root() / "runtime" / "anima_lora").resolve()
    if not (engine / "train.py").is_file():
        message = f"Anima V2 학습 엔진을 찾을 수 없습니다: {engine / 'train.py'}"
        print(f"[Anima V2] {message}")
        raise FileNotFoundError(message)

    source = Path(builtins["TRAIN_DIR"])
    subsets = sorted(path for path in source.iterdir() if path.is_dir())
    images = ensure_dir(run_dir / "resized")
    cache = ensure_dir(run_dir / "embeddings")
    dataset_path = run_dir / "dataset.toml"
    dataset = "[general]\n[[datasets]]\nbatch_size = 1\n"
    for index, subset in enumerate(subsets):
        dataset += (
            "[[datasets.subsets]]\n"
            f"image_dir = {_format_toml_value(str(images / str(index)))}\n"
            f"cache_dir = {_format_toml_value(str(cache / str(index)))}\n"
            "num_repeats = 1\nrecursive = true\n"
        )
    dataset_path.write_text(dataset, encoding="utf-8")

    model_paths = {slot.slot_type: slots[slot.name].replacement for slot in profile.slots}
    defaults = tomllib.loads(profile.config)
    config = {
        "pretrained_model_name_or_path": model_paths["MODEL"],
        "qwen3": model_paths["CLIP"],
        "vae": model_paths["VAE"],
        "dataset_config": str(dataset_path),
        "output_dir": builtins["OUTPUT_DIR"],
        "output_name": builtins["OUTPUT_NAME"],
        "progress_jsonl": "off",
        "network_module": "networks.lora_anima",
        "network_train_unet_only": True,
        "max_train_steps": options.steps_override or defaults["max_train_steps"],
        "learning_rate": options.learning_rate_override or defaults["learning_rate"],
        "network_dim": options.network_dim_override or defaults["network_dim"],
        "network_alpha": options.network_alpha_override or defaults["network_alpha"],
        "seed": options.seed_override if options.seed_override >= 0 else defaults["seed"],
        "optimizer_type": "AdamW",
        "lr_scheduler": "constant",
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "save_model_as": "safetensors",
        "max_data_loader_n_workers": 0,
        "persistent_data_loader_workers": False,
        "use_vae_cache": True,
        "use_text_cache": True,
        "cache_llm_adapter_outputs": True,
        "gradient_checkpointing": options.gradient_checkpointing,
        "torch_compile": v2_options.get("torch_compile", True),
        "compile_dynamic_seq": v2_options.get("torch_compile", True),
        "dynamo_backend": "inductor",
        "activation_memory_budget": v2_options.get("activation_memory_budget", 0.99),
        "attn_mode": v2_options.get("attn_mode", "torch"),
        "use_timestep_mask": v2_options.get("use_timestep_mask", False),
        "min_rank": 1,
        "alpha_rank_scale": 1.0,
        "train_adaln": False,
        "channel_scaling_alpha": 0.0,
        "stream": "bf16",
        "vocab_pack": "",
        "use_repa": False,
        "use_cmmd": False,
        "caption_dropout_rate": 0.0,
        "use_shuffled_caption_variants": False,
        "config_snapshot": True,
    }
    if options.save_every_n_steps > 0:
        config["save_every_n_steps"] = options.save_every_n_steps
    config_path = run_dir / "train.toml"
    config_path.write_text(
        "".join(f"{key} = {_format_toml_value(value)}\n" for key, value in config.items()),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(ensure_dir(plugin_root() / "runtime" / "anima_compile_cache"))
    env["TRITON_CACHE_DIR"] = str(ensure_dir(plugin_root() / "runtime" / "anima_triton_cache"))
    # Do not inherit sd-scripts' library package into this separate process.
    env["PYTHONPATH"] = os.pathsep.join((str(engine), str(plugin_root() / "runtime" / "anime_tools")))
    resolution = options.resolution_override.strip() or defaults.get("resolution", "1024,1024")
    dimensions = tuple(int(part) for part in str(resolution).split(","))
    if len(dimensions) == 1:
        dimensions *= 2
    max_width, max_height = dimensions
    max_area = max_width * max_height
    bucket_step = defaults.get("bucket_reso_steps", 64)
    resized_images = []
    commands = []
    for index, subset in enumerate(subsets):
        destination = ensure_dir(images / str(index))
        for source_image in sorted(path for path in subset.rglob("*")
                                   if path.suffix.lower() in IMAGE_EXTENSIONS):
            try:
                with Image.open(source_image) as original:
                    pixels = original.convert("RGB")
                width, height = pixels.size
                resized_size = (width, height)
                if width * height > max_area:
                    # Match sd-scripts BucketManager.select_bucket(no_upscale=True),
                    # including its aspect-error choice and half-up grid rounding.
                    aspect = width / height
                    scaled_width = math.sqrt(max_area * aspect)
                    scaled_height = max_area / scaled_width
                    rounded_width = int(scaled_width + 0.5) // bucket_step * bucket_step
                    height_from_width = int(rounded_width / aspect + 0.5) // bucket_step * bucket_step
                    rounded_height = int(scaled_height + 0.5) // bucket_step * bucket_step
                    width_from_height = int(rounded_height * aspect + 0.5) // bucket_step * bucket_step
                    if abs(rounded_width / height_from_width - aspect) < abs(width_from_height / rounded_height - aspect):
                        resized_size = (rounded_width, int(rounded_width / aspect + 0.5))
                    else:
                        resized_size = (int(rounded_height * aspect + 0.5), rounded_height)
                    pixels = Image.fromarray(cv2.resize(np.asarray(pixels), resized_size,
                                                       interpolation=cv2.INTER_AREA))
                bucket_width, bucket_height = (size // bucket_step * bucket_step for size in resized_size)
                left = (resized_size[0] - bucket_width) // 2
                top = (resized_size[1] - bucket_height) // 2
                pixels = pixels.crop((left, top, left + bucket_width, top + bucket_height))
                output_image = destination / source_image.relative_to(subset).with_suffix(".png")
                ensure_dir(output_image.parent)
                pixels.save(output_image)
                resized_images.append({"source": str(source_image), "output": str(output_image),
                                       "original_size": [width, height], "resized_size": resized_size,
                                       "bucket_size": [bucket_width, bucket_height]})
                print(f"[Anima V2] resolution={resolution}, no_upscale=True: "
                      f"{source_image.name} {width}x{height} -> {bucket_width}x{bucket_height}")
            except Exception:
                print(f"[Anima V2] image preprocessing failed: {source_image}, resolution={resolution}")
                traceback.print_exc()
                raise
        # Cache and training remain on the engine's public entry points. The
        # upstream tier resize would upscale/rebucket these V1-compatible pixels.
        commands.extend([
            [sys.executable, "scripts/preprocess/cache_latents.py", "--dir", str(images / str(index)),
             "--cache_dir", str(cache / str(index)), "--vae", model_paths["VAE"], "--recursive"],
            [sys.executable, "scripts/preprocess/cache_text_embeddings.py", "--dir", str(subset),
             "--cache_dir", str(cache / str(index)), "--qwen3", model_paths["CLIP"],
             "--dit", model_paths["MODEL"], "--vocab_pack", "", "--recursive"],
        ])
    for command in commands:
        print(f"[Anima V2] preprocessing: {command}")
        run_command(command, cwd=engine, log_path=log_path, env=env)
    command = [sys.executable, "train.py", "--config_file", str(config_path),
               "--max_train_steps", str(config["max_train_steps"])]
    (run_dir / "commands.json").write_text(
        json.dumps({"python": sys.executable, "cwd": str(engine), "resize": resized_images, "preprocess": commands,
                    "training": command}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return config_path, (command, engine, env)
