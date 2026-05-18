from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io

from .profiles import ProfileDefinition, SlotSpec, load_profiles, profile_map, profiles_fingerprint, replace_profile_tokens
from .runtime import (
    IMAGE_EXTENSIONS,
    ensure_dir,
    ensure_sd_scripts_environment,
    export_images,
    get_runtime_paths,
    hash_directory_images,
    hash_tensor_batch,
    hash_text,
    latest_safetensors,
    move_into_artifacts,
    read_caption_files,
    resolve_sd_scripts_file,
    run_command,
    venv_python,
    write_json,
)
import torch
from .sampling import generate_preview, patch_lora_onto_models


NODE_VERSION = "0.1.4"
MAX_TRAIN_STEPS_PATTERN = re.compile(r"^\s*max_train_steps\s*=\s*(\d+)\s*$", re.MULTILINE)
MIXED_PRECISION_PATTERN = re.compile(r'^\s*mixed_precision\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
_TRAIN_STEP_RE = re.compile(r'\|\s*(\d+)/(\d+)\s')


def _notify_phase(text: str) -> None:
    """Send a phase/status text to display on the executing node in ComfyUI."""
    try:
        from server import PromptServer
        server = PromptServer.instance
        if server is not None:
            node_id = getattr(server, 'last_node_id', None)
            if node_id is not None:
                server.send_progress_text(text, node_id)
    except Exception:
        pass


def _send_ws_progress(phase: str, **kwargs) -> None:
    """Send structured progress data to all WebSocket clients via ComfyUI PromptServer."""
    try:
        from server import PromptServer
        server = PromptServer.instance
        if server is not None:
            data = {"phase": phase, **kwargs}
            server.send_sync("md_soya_progress", data)
    except Exception:
        pass


@io.comfytype(io_type="CONTEXT")
class ContextType(io.ComfyTypeIO):
    Type = dict


@io.comfytype(io_type="TAGGING_OPTIONS")
class TaggingOptionsType(io.ComfyTypeIO):
    Type = dict


@io.comfytype(io_type="TRAIN_OPTIONS")
class TrainOptionsType(io.ComfyTypeIO):
    Type = dict


@dataclass(frozen=True)
class ResolvedSlot:
    name: str
    replacement: str
    fingerprint: str


@dataclass(frozen=True)
class TaggingOptions:
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    prepend_tags: str = ""
    append_tags: str = ""
    exclude_tags: str = ""
    replace_tags: str = ""
    remove_underscore: bool = True


@dataclass(frozen=True)
class TrainOptions:
    steps_override: int = 0
    learning_rate_override: float = 0.0
    network_dim_override: int = 0
    network_alpha_override: int = 0
    resolution_override: str = ""
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = True
    train_text_encoder: bool = False
    text_encoder_lr: float = 0.0
    save_every_n_steps: int = 0
    seed_override: int = -1
    force_retrain: bool = False


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _last_lora_info_path() -> Path:
    return get_runtime_paths().root / "last_lora.json"


def _hash_options(value: dict[str, Any]) -> str:
    return hash_text(json.dumps(value, sort_keys=True))


def _split_tags(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_prompt_groups(text: str) -> list[str]:
    # Format: [1]tags for image 1\n[2]tags for image 2\n...
    # Split on [digit(s)] markers and return the text after each marker
    parts = re.split(r'\[\d+\]', text)
    return [p.strip() for p in parts if p.strip()]


def _tagging_options_from_input(value: Any | None) -> TaggingOptions:
    if not value:
        return TaggingOptions()
    return TaggingOptions(
        general_threshold=float(value.get("general_threshold", 0.35)),
        character_threshold=float(value.get("character_threshold", 0.85)),
        prepend_tags=str(value.get("prepend_tags", "")),
        append_tags=str(value.get("append_tags", "")),
        exclude_tags=str(value.get("exclude_tags", "")),
        replace_tags=str(value.get("replace_tags", "")),
        remove_underscore=bool(value.get("remove_underscore", True)),
    )


def _train_options_from_input(value: Any | None) -> TrainOptions:
    if not value:
        return TrainOptions()
    return TrainOptions(
        steps_override=int(value.get("steps_override", 0)),
        learning_rate_override=float(value.get("learning_rate_override", 0.0)),
        network_dim_override=int(value.get("network_dim_override", 0)),
        network_alpha_override=int(value.get("network_alpha_override", 0)),
        resolution_override=str(value.get("resolution_override", "")),
        gradient_checkpointing=bool(value.get("gradient_checkpointing", True)),
        cache_latents=bool(value.get("cache_latents", True)),
        cache_text_encoder_outputs=bool(value.get("cache_text_encoder_outputs", True)),
        save_every_n_steps=int(value.get("save_every_n_steps", 0)),
        seed_override=int(value.get("seed_override", -1)),
        force_retrain=bool(value.get("force_retrain", False)),
    )


def _tagging_options_fingerprint(options: TaggingOptions) -> str:
    return _hash_options(options.__dict__)


def _train_options_fingerprint(options: TrainOptions) -> str:
    return _hash_options(options.__dict__)


def _effective_max_train_steps(profile: ProfileDefinition, options: TrainOptions) -> int:
    if options.steps_override > 0:
        return options.steps_override
    match = MAX_TRAIN_STEPS_PATTERN.search(profile.config)
    if match:
        return max(1, int(match.group(1)))
    return 50


def _profile_slots_by_type(profile: ProfileDefinition, slot_type: str) -> list[SlotSpec]:
    return [slot for slot in profile.slots if slot.slot_type == slot_type]


def _primary_profile_slot(profile: ProfileDefinition, slot_type: str) -> SlotSpec | None:
    slots = _profile_slots_by_type(profile, slot_type)
    return slots[0] if slots else None


def _recover_model_checkpoint_path(model: Any) -> str:
    cached = getattr(model, "cached_patcher_init", None)
    if not cached or len(cached) < 2 or not cached[1]:
        raise RuntimeError("This MODEL does not expose a recoverable checkpoint path. Load it with a checkpoint loader first.")
    checkpoint_path = cached[1][0]
    if not isinstance(checkpoint_path, str):
        raise RuntimeError("Recovered checkpoint path was not a string.")
    return checkpoint_path


def _recover_clip_paths(clip: Any) -> list[str]:
    patcher = getattr(clip, "patcher", None)
    cached = getattr(patcher, "cached_patcher_init", None)
    if not cached or len(cached) < 2 or not cached[1]:
        raise RuntimeError("This CLIP input does not expose recoverable checkpoint metadata.")
    ckpt_paths = cached[1][0]
    if isinstance(ckpt_paths, (list, tuple)):
        return [str(path) for path in ckpt_paths]
    raise RuntimeError("Recovered CLIP paths were not a list.")


def _resolve_string_slot(name: str, value: str) -> ResolvedSlot:
    return ResolvedSlot(name=name, replacement=value, fingerprint=hash_text(value))


def _resolve_model_slot(name: str, value: Any) -> ResolvedSlot:
    checkpoint_path = _recover_model_checkpoint_path(value)
    return ResolvedSlot(name=name, replacement=checkpoint_path, fingerprint=hash_text(checkpoint_path))


def _export_state_dict_artifact(state_dict: dict[str, Any], suffix: str) -> tuple[str, Path]:
    paths = get_runtime_paths()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=paths.artifacts) as handle:
        temp_path = Path(handle.name)
    comfy.utils.save_torch_file(state_dict, str(temp_path))
    return move_into_artifacts(temp_path, paths.artifacts, suffix)


def _resolve_clip_slot(name: str, value: Any) -> ResolvedSlot:
    try:
        clip_paths = _recover_clip_paths(value)
        fingerprint = hash_text("|".join(clip_paths))
        return ResolvedSlot(name=name, replacement=clip_paths[0], fingerprint=fingerprint)
    except Exception:
        digest, exported_path = _export_state_dict_artifact(value.get_sd(), ".safetensors")
        return ResolvedSlot(name=name, replacement=str(exported_path), fingerprint=digest)


def _resolve_vae_slot(name: str, value: Any) -> ResolvedSlot:
    digest, exported_path = _export_state_dict_artifact(value.get_sd(), ".safetensors")
    return ResolvedSlot(name=name, replacement=str(exported_path), fingerprint=digest)


def _resolve_slot(slot: SlotSpec, raw_value: Any) -> ResolvedSlot:
    if slot.slot_type == "STRING":
        return _resolve_string_slot(slot.name, raw_value)
    if slot.slot_type == "MODEL":
        return _resolve_model_slot(slot.name, raw_value)
    if slot.slot_type == "CLIP":
        return _resolve_clip_slot(slot.name, raw_value)
    if slot.slot_type == "VAE":
        return _resolve_vae_slot(slot.name, raw_value)
    raise RuntimeError(f"Unsupported slot type: {slot.slot_type}")


def _tag_dataset(paths, dataset_dir: Path, log_path: Path, options: TaggingOptions) -> None:
    if any(dataset_dir.glob("*.txt")):
        return
    ensure_sd_scripts_environment(paths, log_path=log_path)
    python_path = venv_python(paths.venv)
    tagger_script = resolve_sd_scripts_file(paths, "tag_images_by_wd14_tagger.py")
    tagger_model_dir = paths.sd_scripts / "wd14_tagger_model"
    onnx_model_path = tagger_model_dir / "SmilingWolf_wd-v1-4-convnext-tagger-v2" / "model.onnx"
    command = [
        str(python_path),
        str(tagger_script),
        "--batch_size",
        "1",
        "--caption_extension",
        ".txt",
        "--general_threshold",
        str(options.general_threshold),
        "--character_threshold",
        str(options.character_threshold),
        "--model_dir",
        str(tagger_model_dir),
        "--onnx",
        "--recursive",
        str(dataset_dir),
    ]
    if options.remove_underscore:
        command.append("--remove_underscore")
    if options.exclude_tags.strip():
        command.extend(["--undesired_tags", options.exclude_tags])
    if options.replace_tags.strip():
        command.extend(["--tag_replacement", options.replace_tags])
    if not onnx_model_path.exists():
        command.append("--force_download")
    run_command(command, cwd=paths.sd_scripts, log_path=log_path)


def _apply_caption_options(dataset_dir: Path, options: TaggingOptions) -> dict[str, str]:
    captions = read_caption_files(dataset_dir)
    prepend_tags = _split_tags(options.prepend_tags)
    append_tags = _split_tags(options.append_tags)
    exclude_tags = set(_split_tags(options.exclude_tags))
    for path in sorted(dataset_dir.glob("*.txt")):
        tags = _split_tags(captions.get(path.stem, ""))
        filtered_tags = [tag for tag in tags if tag not in exclude_tags]
        final_tags: list[str] = []
        for tag in [*prepend_tags, *filtered_tags, *append_tags]:
            if tag and tag not in final_tags:
                final_tags.append(tag)
        path.write_text(", ".join(final_tags), encoding="utf-8")
    return read_caption_files(dataset_dir)


def _prepare_dataset(
    images: Any,
    log_path: Path,
    options: TaggingOptions,
    target_steps: int,
    source_dir: Path | None = None,
    custom_tags: str = "",
    per_image_tags: list[dict] | None = None,
) -> tuple[Path, str, str, dict[str, str]]:
    paths = get_runtime_paths()
    
    if source_dir:
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(source_dir.glob(f"*{ext}"))
            image_files.extend(source_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            raise RuntimeError(f"No supported image files found in {source_dir}")

        image_count = len(image_files)
        image_hash = hash_directory_images(source_dir)
    else:
        image_hash = hash_tensor_batch(images)
        image_count = images.shape[0] if hasattr(images, "shape") and len(images.shape) > 0 else 1

    dataset_key = hash_text(f"{image_hash}|{_tagging_options_fingerprint(options)}|steps:{target_steps}")
    dataset_dir = ensure_dir(paths.datasets / dataset_key)
    repeat_count = max(1, -(-target_steps // max(1, int(image_count))))
    train_subset_dir = ensure_dir(dataset_dir / f"{repeat_count}_reference")

    if source_dir:
        # Copy files to the structured dataset directory
        for img_path in image_files:
            shutil.copy2(img_path, train_subset_dir / img_path.name)
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                shutil.copy2(txt_path, train_subset_dir / txt_path.name)
    else:
        export_images(images, train_subset_dir)

    if per_image_tags:
        for entry in per_image_tags:
            idx = entry["index"]
            img_file = train_subset_dir / f"image_{idx:03d}.png"
            if img_file.exists():
                caption_path = img_file.with_suffix(".txt")
                caption_path.write_text(entry["positive_tags"].strip(), encoding="utf-8")
    elif custom_tags and custom_tags.strip():
        # Write custom tags directly, skip WD14 tagging
        tag_text = custom_tags.strip()
        for img_file in sorted(train_subset_dir.iterdir()):
            if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                caption_path = img_file.with_suffix(".txt")
                caption_path.write_text(tag_text, encoding="utf-8")
    else:
        if source_dir:
            _tag_dataset(paths, source_dir, log_path=log_path, options=options)
            _apply_caption_options(source_dir, options)
            # Re-copy updated caption files to train_subset_dir
            for img_path in image_files:
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    shutil.copy2(txt_path, train_subset_dir / txt_path.name)
        else:
            _tag_dataset(paths, train_subset_dir, log_path=log_path, options=options)
            _apply_caption_options(train_subset_dir, options)

    captions = read_caption_files(train_subset_dir)
    if not captions:
        raise RuntimeError("Dataset preparation did not produce any caption files.")
        
    caption_digest = hashlib.blake2b(digest_size=4)
    for key, value in sorted(captions.items()):
        caption_digest.update(key.encode("utf-8"))
        caption_digest.update(value.encode("utf-8"))
        
    return dataset_dir, image_hash, caption_digest.hexdigest(), captions


def _merge_run_log(temp_log: Path, final_log: Path) -> None:
    if not temp_log.exists():
        return
    ensure_dir(final_log.parent)
    content = temp_log.read_text(encoding="utf-8")
    with final_log.open("a", encoding="utf-8") as handle:
        handle.write(content)
    temp_log.unlink(missing_ok=True)
    temp_dir = temp_log.parent
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()


def _cache_key(
    checkpoint_path: str,
    profile: ProfileDefinition,
    image_hash: str,
    captions_hash: str,
    slots: dict[str, ResolvedSlot],
    tagging_options: TaggingOptions,
    train_options: TrainOptions,
) -> str:
    digest = hashlib.blake2b(digest_size=4)
    digest.update(NODE_VERSION.encode("utf-8"))
    digest.update(checkpoint_path.encode("utf-8"))
    digest.update(profile.file_hash.encode("utf-8"))
    digest.update(image_hash.encode("utf-8"))
    digest.update(captions_hash.encode("utf-8"))
    digest.update(_tagging_options_fingerprint(tagging_options).encode("utf-8"))
    digest.update(_train_options_fingerprint(train_options).encode("utf-8"))
    for name in sorted(slots):
        digest.update(name.encode("utf-8"))
        digest.update(slots[name].fingerprint.encode("utf-8"))
    return digest.hexdigest()


def _builtins_for_run(
    dataset_dir: Path,
    output_dir: Path,
    output_name: str,
) -> dict[str, str]:
    return {
        "TRAIN_DIR": str(dataset_dir),
        "OUTPUT_DIR": str(output_dir),
        "OUTPUT_NAME": output_name,
        "CAPTION_EXTENSION": ".txt",
    }


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    return json.dumps(str(value))


def _set_toml_key(config_text: str, key: str, value: Any) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    replacement = f"{key} = {_format_toml_value(value)}"
    if pattern.search(config_text):
        return pattern.sub(replacement, config_text, count=1)
    if not config_text.endswith("\n"):
        config_text += "\n"
    return config_text + replacement + "\n"


def _apply_train_options(config_text: str, options: TrainOptions) -> str:
    overrides: dict[str, Any] = {}
    if options.steps_override > 0:
        overrides["max_train_steps"] = options.steps_override
    if options.learning_rate_override > 0:
        overrides["learning_rate"] = options.learning_rate_override
    if options.network_dim_override > 0:
        overrides["network_dim"] = options.network_dim_override
    if options.network_alpha_override > 0:
        overrides["network_alpha"] = options.network_alpha_override
    if options.resolution_override.strip():
        overrides["resolution"] = options.resolution_override.strip()
    overrides["gradient_checkpointing"] = options.gradient_checkpointing
    overrides["cache_latents"] = options.cache_latents
    if options.train_text_encoder:
        # Remove network_train_unet_only so TE is also trained
        config_text = re.sub(r"(?m)^network_train_unet_only\s*=.*\n?", "", config_text)
        overrides["cache_text_encoder_outputs"] = False
        if options.text_encoder_lr > 0:
            overrides["text_encoder_lr"] = options.text_encoder_lr
    else:
        overrides["cache_text_encoder_outputs"] = options.cache_text_encoder_outputs
    if options.save_every_n_steps > 0:
        overrides["save_every_n_steps"] = options.save_every_n_steps
    if os.name == "nt":
        overrides["max_data_loader_n_workers"] = 0
        overrides["persistent_data_loader_workers"] = False
    if options.seed_override >= 0:
        overrides["seed"] = options.seed_override
    for key, value in overrides.items():
        config_text = _set_toml_key(config_text, key, value)
    return config_text


def _accelerate_mixed_precision(config_path: Path) -> str | None:
    match = MIXED_PRECISION_PATTERN.search(config_path.read_text(encoding="utf-8"))
    if not match:
        return None
    value = match.group(1).strip().lower()
    if value in {"fp16", "bf16", "fp8", "no"}:
        return value
    return None


def _write_resolved_config(
    profile: ProfileDefinition,
    slot_values: dict[str, ResolvedSlot],
    builtins: dict[str, str],
    run_dir: Path,
    train_options: TrainOptions,
) -> Path:
    config_path = run_dir / "config.toml"
    rendered = replace_profile_tokens(
        profile.config,
        {name: slot.replacement for name, slot in slot_values.items()},
        builtins,
    )
    rendered = _apply_train_options(rendered, train_options)
    config_path.write_text(rendered, encoding="utf-8")
    return config_path


def _run_training(profile: ProfileDefinition, run_dir: Path, output_dir: Path, config_path: Path, log_path: Path, total_steps: int | None = None) -> Path:
    paths = get_runtime_paths()
    ensure_sd_scripts_environment(paths, log_path=log_path)
    python_path = venv_python(paths.venv)
    training_script = resolve_sd_scripts_file(paths, profile.script)
    command = [str(python_path)]
    if os.name == "nt":
        command.extend(
            [
                "-m",
                "accelerate.commands.launch",
                "--num_cpu_threads_per_process",
                "2",
            ]
        )
        mixed_precision = _accelerate_mixed_precision(config_path)
        if mixed_precision is not None:
            command.extend(["--mixed_precision", mixed_precision])
    command.extend([str(training_script), "--config_file", str(config_path)])

    pbar = comfy.utils.ProgressBar(total_steps) if total_steps and total_steps > 0 else None
    t_train_start = time.monotonic()

    def on_line(text):
        if pbar is not None:
            match = _TRAIN_STEP_RE.search(text)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                pbar.update_absolute(current, total)
                _notify_phase(f"Training step {current}/{total}")
                # 구조화된 진행률 WebSocket 전송
                elapsed = time.monotonic() - t_train_start
                if current > 0:
                    avg_s = elapsed / current
                    remain_s = avg_s * (total - current)
                    remain_min = remain_s / 60
                else:
                    remain_min = 0
                _send_ws_progress("training",
                    step=current, total=total,
                    elapsed_sec=round(elapsed, 1),
                    remaining_min=round(remain_min, 1))

    run_command(command, cwd=paths.sd_scripts, log_path=log_path, line_callback=on_line)
    trained_lora = latest_safetensors(output_dir)
    if trained_lora is None:
        raise RuntimeError(f"Training completed but no LoRA file was found in {output_dir}")
    return trained_lora


def _record_last_lora(lora_path: Path) -> None:
    write_json(
        _last_lora_info_path(),
        {
            "path": str(lora_path),
        },
    )


def _execute_reference_lora(
    model, clip, profile,
    model_strength=1.0, clip_strength=1.0,
    tagging_options=None, train_options=None,
    save_path="", vae=None, context=None,
    preview_enable=False, preview_prompt_index=0, preview_seed=42,
    preview_steps=20, preview_cfg=7.0, preview_sampler="euler", preview_scheduler="normal",
    preview_width=512, preview_height=512,
    preview_positive_prompt="", preview_negative_prompt="",
    preview_no_lora=False,
    preview_model=None, preview_clip=None, preview_vae=None,
) -> tuple:
    print("[md_soya] _execute_reference_lora called")
    # Fallback: use training model/clip/vae for preview if separate preview inputs not provided
    pv_model = preview_model if preview_model is not None else model
    pv_clip = preview_clip if preview_clip is not None else clip
    pv_vae = preview_vae if preview_vae is not None else vae
    # Support both string (profile key) and dict (legacy combo format)
    if isinstance(profile, str):
        profile_key = profile
        slot_values = {}
    else:
        profile_key = profile.get("profile", "")
        slot_values = {k: v for k, v in profile.items() if k != "profile"}

    if vae is not None:
        slot_values["vae"] = vae

    if context is None:
        raise RuntimeError("context input is required. Connect a Context Builder node.")
    images = context["images"]
    per_image_tags = context["entries"]
    print(f"[md_soya] images shape={getattr(images, 'shape', 'N/A')}, per_image_tags count={len(per_image_tags)}")
    for entry in per_image_tags:
        print(f"[md_soya]   image {entry['index']}: positive={repr(entry.get('positive_tags', ''))}, negative={repr(entry.get('negative_tags', ''))}")

    all_profiles = profile_map(_plugin_root())
    print(f"[md_soya] available profiles: {list(all_profiles.keys())}")
    if profile_key not in all_profiles:
        raise RuntimeError(f"Profile '{profile_key}' not found. Available: {list(all_profiles.keys())}")
    selected_profile = all_profiles[profile_key]
    print(f"[md_soya] selected profile: {selected_profile.name} (key={selected_profile.key})")
    resolved_tagging = _tagging_options_from_input(tagging_options)
    resolved_train = _train_options_from_input(train_options)
    target_steps = _effective_max_train_steps(selected_profile, resolved_train)
    print(f"[md_soya] target_steps={target_steps}")
    model_slot = _primary_profile_slot(selected_profile, "MODEL")
    if model_slot is None:
        raise RuntimeError(f"Profile '{selected_profile.name}' must define a MODEL slot such as '{{{{model:MODEL}}}}'.")
    checkpoint_path = _recover_model_checkpoint_path(model)
    print(f"[md_soya] checkpoint_path={checkpoint_path}")

    temp_image_hash = hash_tensor_batch(images)
    print(f"[md_soya] image_hash={temp_image_hash}")

    paths = get_runtime_paths()
    print(f"[md_soya] runtime root={paths.root}, sd_scripts={paths.sd_scripts}")
    temp_run_dir = ensure_dir(paths.cache / f"_inflight_{temp_image_hash}")
    temp_run_log = temp_run_dir / "run.log"
    print("[md_soya] preparing dataset...")
    _notify_phase("Preparing dataset...")
    dataset_dir, image_hash, captions_hash, captions = _prepare_dataset(
        images,
        log_path=temp_run_log,
        options=resolved_tagging,
        target_steps=target_steps,
        per_image_tags=per_image_tags,
    )
    print(f"[md_soya] dataset prepared: {dataset_dir}, captions count={len(captions)}")
    for name, caption in sorted(captions.items()):
        print(f"[md_soya]   caption '{name}': {repr(caption)}")

    all_tags_text = "\n".join([v for k, v in captions.items()])

    resolved_slots: dict[str, ResolvedSlot] = {}
    for slot in selected_profile.slots:
        if slot.slot_type == "MODEL":
            resolved_slots[slot.name] = _resolve_slot(slot, model)
            continue
        if slot.slot_type == "CLIP":
            resolved_slots[slot.name] = _resolve_slot(slot, clip)
            continue
        if slot.name not in slot_values:
            raise RuntimeError(f"Profile '{selected_profile.name}' requires input '{slot.name}'.")
        resolved_slots[slot.name] = _resolve_slot(slot, slot_values[slot.name])
    print(f"[md_soya] resolved_slots: {list(resolved_slots.keys())}")

    cache_key = _cache_key(
        checkpoint_path=checkpoint_path,
        profile=selected_profile,
        image_hash=image_hash,
        captions_hash=captions_hash,
        slots=resolved_slots,
        tagging_options=resolved_tagging,
        train_options=resolved_train,
    )
    print(f"[md_soya] cache_key={cache_key}")

    run_dir = ensure_dir(paths.cache / cache_key)
    run_log = run_dir / "run.log"
    if temp_run_log != run_log:
        _merge_run_log(temp_run_log, run_log)
    save_path_clean = save_path.strip().strip("/\\")
    if not save_path_clean:
        raise RuntimeError("save_path is required. Enter a path like 'hamin/anima-01'.")
    date_folder = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = ensure_dir(paths.outputs / save_path_clean / date_folder)
    output_name = secrets.token_hex(4)
    manifest = run_dir / "manifest.json"

    print("[md_soya] starting training...")
    _notify_phase(f"Training ({target_steps} steps)...")
    _send_ws_progress("preparing", message=f"Training ({target_steps} steps)...")
    ensure_sd_scripts_environment(paths, log_path=run_log)
    builtins = _builtins_for_run(dataset_dir, output_dir, output_name)
    config_path = _write_resolved_config(selected_profile, resolved_slots, builtins, run_dir, resolved_train)
    print(f"[md_soya] config written to {config_path}")
    write_json(
        manifest,
        {
            "cache_key": cache_key,
            "checkpoint_path": checkpoint_path,
            "profile": selected_profile.key,
            "profile_file": str(selected_profile.file_path),
            "captions": captions,
            "tagging_options": resolved_tagging.__dict__,
            "train_options": resolved_train.__dict__,
            "resolved_slots": {name: slot.replacement for name, slot in resolved_slots.items()},
            "config_path": str(config_path),
        },
    )
    # --- preview auto-save_every_n_steps injection ---
    if preview_enable and resolved_train.save_every_n_steps <= 0:
        auto_save_every = max(1, target_steps // 4)
        config_text = config_path.read_text(encoding="utf-8")
        config_text = _set_toml_key(config_text, "save_every_n_steps", auto_save_every)
        config_path.write_text(config_text, encoding="utf-8")
        print(f"[md_soya] preview enabled, auto save_every_n_steps={auto_save_every}")

    comfy.model_management.unload_all_models()
    soft_empty_cache = getattr(comfy.model_management, "soft_empty_cache", None)
    if callable(soft_empty_cache):
        soft_empty_cache()
    print("[md_soya] calling _run_training...")
    trained_lora = _run_training(selected_profile, run_dir, output_dir, config_path, log_path=run_log, total_steps=target_steps)
    print(f"[md_soya] training done, lora={trained_lora}")
    _notify_phase("Training complete!")
    _send_ws_progress("training_complete", message="Training complete!", lora_path=str(trained_lora))

    # --- Post-hoc preview: sample all checkpoints after training ---
    preview_images = []
    preview_map: dict[Path, list] = {}  # ckpt_path -> [(prompt_index, img_tensor), ...]
    parsed_pos = _parse_prompt_groups(preview_positive_prompt) if preview_positive_prompt else []
    parsed_neg = _parse_prompt_groups(preview_negative_prompt) if preview_negative_prompt else []

    if preview_enable and pv_vae is not None:
        try:
            checkpoints = sorted(output_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
            num_prompts_per_ckpt = len(parsed_pos) if parsed_pos else 1
            total_previews = len(checkpoints) * num_prompts_per_ckpt
            print(f"[md_soya] Post-hoc preview: found {len(checkpoints)} checkpoints, {num_prompts_per_ckpt} prompts each = {total_previews} total")
            _notify_phase(f"Preview 0/{total_previews}...")
            _send_ws_progress("preview_start", total=total_previews)
            t_preview_start = time.monotonic()
            for ckpt in checkpoints:
                print(f"[md_soya] Sampling preview for {ckpt.name}...")
                try:
                    comfy.model_management.unload_all_models()
                    if callable(soft_empty_cache):
                        soft_empty_cache()

                    m_patched, c_patched = patch_lora_onto_models(
                        pv_model, pv_clip, str(ckpt), model_strength, clip_strength
                    )

                    w = preview_width if preview_width > 0 else (context["images"].shape[2] if "images" in context else 512)
                    h = preview_height if preview_height > 0 else (context["images"].shape[1] if "images" in context else 512)

                    preview_map[ckpt] = []
                    if parsed_pos:
                        # Multi-prompt mode: generate one preview per prompt group
                        for pi, pos_text in enumerate(parsed_pos):
                            neg_text = parsed_neg[pi] if pi < len(parsed_neg) else (parsed_neg[-1] if parsed_neg else "")
                            img = generate_preview(
                                m_patched, c_patched, pv_vae,
                                pos_text, neg_text,
                                w, h,
                                preview_seed + len(preview_images),
                                preview_steps, preview_cfg,
                                preview_sampler, preview_scheduler,
                            )
                            preview_images.append(img)
                            preview_map[ckpt].append((pi + 1, img))
                            done = len(preview_images)
                            elapsed = time.monotonic() - t_preview_start
                            avg_s = elapsed / done
                            remain_s = avg_s * (total_previews - done)
                            remain_m = remain_s / 60
                            _notify_phase(f"Preview {done}/{total_previews} (~{remain_m:.1f}min left)")
                            _send_ws_progress("preview",
                                current=done, total=total_previews,
                                remaining_min=round(remain_m, 1),
                                checkpoint=ckpt.name)
                            print(f"[md_soya] Preview done for {ckpt.name} [{pi+1}/{len(parsed_pos)}] ({done}/{total_previews}): {img.shape}")
                    else:
                        # Legacy single-prompt mode
                        entries = context.get("entries", [])
                        idx = min(preview_prompt_index, len(entries) - 1) if entries else 0
                        entry = entries[idx] if entries else {"positive_tags": "", "negative_tags": ""}
                        img = generate_preview(
                            m_patched, c_patched, pv_vae,
                            entry.get("positive_tags", ""), entry.get("negative_tags", ""),
                            w, h,
                            preview_seed + len(preview_images),
                            preview_steps, preview_cfg,
                            preview_sampler, preview_scheduler,
                        )
                        preview_images.append(img)
                        preview_map[ckpt].append((0, img))
                        done = len(preview_images)
                        elapsed = time.monotonic() - t_preview_start
                        avg_s = elapsed / done
                        remain_s = avg_s * (total_previews - done)
                        remain_m = remain_s / 60
                        _notify_phase(f"Preview {done}/{total_previews} (~{remain_m:.1f}min left)")
                        _send_ws_progress("preview",
                            current=done, total=total_previews,
                            remaining_min=round(remain_m, 1),
                            checkpoint=ckpt.name)
                        print(f"[md_soya] Preview done for {ckpt.name} ({done}/{total_previews}): {img.shape}")
                except Exception as exc:
                    print(f"[md_soya] Preview failed for {ckpt.name}: {exc}")
                    continue
                finally:
                    del m_patched, c_patched
                    comfy.model_management.unload_all_models()
        except Exception as exc:
            print(f"[md_soya] Post-hoc preview error: {exc}")

    # --- No-LoRA comparison previews (unpatched preview model/clip) ---
    no_lora_images = []  # [(prompt_index, img_tensor), ...]
    if preview_no_lora and preview_enable and pv_vae is not None:
        try:
            comfy.model_management.unload_all_models()
            if callable(soft_empty_cache):
                soft_empty_cache()

            w = preview_width if preview_width > 0 else (context["images"].shape[2] if "images" in context else 512)
            h = preview_height if preview_height > 0 else (context["images"].shape[1] if "images" in context else 512)

            if parsed_pos:
                for pi, pos_text in enumerate(parsed_pos):
                    neg_text = parsed_neg[pi] if pi < len(parsed_neg) else (parsed_neg[-1] if parsed_neg else "")
                    img = generate_preview(
                        pv_model, pv_clip, pv_vae,
                        pos_text, neg_text,
                        w, h,
                        preview_seed + len(preview_images),
                        preview_steps, preview_cfg,
                        preview_sampler, preview_scheduler,
                    )
                    no_lora_images.append((pi + 1, img))
                    preview_images.append(img)
                    print(f"[md_soya] No-LoRA preview [{pi+1}/{len(parsed_pos)}] done: {img.shape}")
            else:
                entries = context.get("entries", [])
                idx = min(preview_prompt_index, len(entries) - 1) if entries else 0
                entry = entries[idx] if entries else {"positive_tags": "", "negative_tags": ""}
                img = generate_preview(
                    pv_model, pv_clip, pv_vae,
                    entry.get("positive_tags", ""), entry.get("negative_tags", ""),
                    w, h,
                    preview_seed + len(preview_images),
                    preview_steps, preview_cfg,
                    preview_sampler, preview_scheduler,
                )
                no_lora_images.append((0, img))
                preview_images.append(img)
                print(f"[md_soya] No-LoRA preview done: {img.shape}")

            comfy.model_management.unload_all_models()
        except Exception as exc:
            print(f"[md_soya] No-LoRA preview error: {exc}")

    # 전체 완료
    _send_ws_progress("all_complete", message="All done!", lora_path=str(trained_lora))

    # --- Save preview JPGs, config TOML, and JSON mapping for each checkpoint ---
    import gc
    gc.collect()
    from PIL import Image as PILImage

    no_lora_filenames = []
    for ckpt_path in sorted(output_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime):
        base = ckpt_path.stem  # e.g. "7cfdab8d" or "7cfdab8d-step00000025"

        previews_for_ckpt = preview_map.get(ckpt_path, [])
        preview_filenames = []
        for pidx, img_tensor in previews_for_ckpt:
            if pidx > 0:
                jpg_name = f"{base}-{pidx}.jpg"
            else:
                jpg_name = f"{base}.jpg"
            img_np = (img_tensor[0].cpu().numpy() * 255).astype("uint8")
            PILImage.fromarray(img_np).save(str(output_dir / jpg_name))
            preview_filenames.append(jpg_name)

        shutil.copy2(config_path, output_dir / f"{base}.toml")

        write_json(
            output_dir / f"{base}.json",
            {
                "lora_file": ckpt_path.name,
                "config_file": f"{base}.toml",
                "previews": preview_filenames,
            },
        )

    # Save no-LoRA comparison images using trained_lora's base name, appended after existing preview indices
    if no_lora_images and trained_lora is not None:
        final_base = trained_lora.stem
        # Reload final checkpoint JSON to append
        final_json_path = output_dir / f"{final_base}.json"
        final_json = json.loads(final_json_path.read_text(encoding="utf-8")) if final_json_path.exists() else {}
        final_preview_filenames = final_json.get("previews", [])
        next_idx = len(final_preview_filenames) + 1
        for pidx, img_tensor in no_lora_images:
            jpg_name = f"{final_base}-{next_idx}.jpg"
            img_np = (img_tensor[0].cpu().numpy() * 255).astype("uint8")
            PILImage.fromarray(img_np).save(str(output_dir / jpg_name))
            no_lora_filenames.append(jpg_name)
            final_preview_filenames.append(jpg_name)
            next_idx += 1
        final_json["previews"] = final_preview_filenames
        write_json(final_json_path, final_json)

    _record_last_lora(trained_lora)

    info = f"[trained] {trained_lora.name} -> {trained_lora}"
    if all_tags_text:
        info += f"\ntags: {all_tags_text}"

    if preview_images:
        preview_batch = torch.cat(preview_images, dim=0)
    else:
        preview_batch = None
    return info, preview_batch


def _profile_choice_inputs(profile: ProfileDefinition) -> list[Any]:
    inputs: list[Any] = []
    for slot in profile.slots:
        if slot.slot_type in {"MODEL", "CLIP"}:
            continue
        if slot.slot_type == "STRING":
            inputs.append(io.String.Input(slot.name, multiline=False))
        elif slot.slot_type == "VAE":
            inputs.append(io.Vae.Input(slot.name))
    return inputs



class MdSoyaInstantReferenceLoRA(io.ComfyNode):
    CATEGORY = "reference/training"

    @classmethod
    def define_schema(cls) -> io.Schema:
        profiles = load_profiles(_plugin_root())
        profile_keys = [profile.key for profile in profiles]
        return io.Schema(
            node_id="md_soya_InstantReferenceLoRA",
            display_name="md_soya Instant Reference LoRA",
            category=cls.CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Float.Input("model_strength", default=1.0),
                io.Float.Input("clip_strength", default=1.0),
                io.String.Input("profile", default=profile_keys[0] if profile_keys else ""),
                ContextType.Input("context"),
                io.Vae.Input("vae", optional=True),
                io.String.Input("save_path", default=""),
                TaggingOptionsType.Input("tagging_options", optional=True),
                TrainOptionsType.Input("train_options", optional=True),
                io.Boolean.Input("preview_enable", default=False),
                io.Int.Input("preview_prompt_index", default=0),
                io.Int.Input("preview_seed", default=42),
                io.Int.Input("preview_steps", default=20),
                io.Float.Input("preview_cfg", default=7.0),
                io.String.Input("preview_sampler", default="euler"),
                io.String.Input("preview_scheduler", default="normal"),
                io.Int.Input("preview_width", default=512),
                io.Int.Input("preview_height", default=512),
                io.String.Input("preview_positive_prompt", default="", multiline=True),
                io.String.Input("preview_negative_prompt", default="", multiline=True),
                io.Boolean.Input("preview_no_lora", default=False),
                io.Model.Input("preview_model", optional=True),
                io.Clip.Input("preview_clip", optional=True),
                io.Vae.Input("preview_vae", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="info"),
                io.Image.Output("preview_images"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        profiles = load_profiles(_plugin_root())
        return profiles_fingerprint(profiles)

    @classmethod
    def execute(cls, model, clip, model_strength=1.0, clip_strength=1.0, profile="", tagging_options=None, train_options=None, save_path="", vae=None, context=None, preview_enable=False, preview_prompt_index=0, preview_seed=42, preview_steps=20, preview_cfg=7.0, preview_sampler="euler", preview_scheduler="normal", preview_width=512, preview_height=512, preview_positive_prompt="", preview_negative_prompt="", preview_model=None, preview_clip=None, preview_vae=None) -> io.NodeOutput:
        info, preview_batch = _execute_reference_lora(
            model,
            clip,
            profile,
            model_strength=model_strength,
            clip_strength=clip_strength,
            tagging_options=tagging_options,
            train_options=train_options,
            save_path=save_path,
            vae=vae,
            context=context,
            preview_enable=preview_enable,
            preview_prompt_index=preview_prompt_index,
            preview_seed=preview_seed,
            preview_steps=preview_steps,
            preview_cfg=preview_cfg,
            preview_sampler=preview_sampler,
            preview_scheduler=preview_scheduler,
            preview_width=preview_width,
            preview_height=preview_height,
            preview_positive_prompt=preview_positive_prompt,
            preview_negative_prompt=preview_negative_prompt,
            preview_model=preview_model,
            preview_clip=preview_clip,
            preview_vae=preview_vae,
        )
        return io.NodeOutput(info, preview_batch)
class ReferenceTrainingExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MdSoyaInstantReferenceLoRA]


# V1 wrapper for NODE_CLASS_MAPPINGS compatibility
class InstantReferenceLoRAV1:
    CATEGORY = "reference/training"
    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("info", "preview_images",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        profiles = load_profiles(_plugin_root())
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "profile": ("STRING", {"default": profiles[0].key if profiles else ""}),
                "context": ("CONTEXT",),
            },
            "optional": {
                "vae": ("VAE",),
                "save_path": ("STRING", {"default": "", "multiline": False}),
                "tagging_options": ("TAGGING_OPTIONS",),
                "train_options": ("TRAIN_OPTIONS",),
                "preview_enable": ("BOOLEAN", {"default": False}),
                "preview_prompt_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "preview_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "preview_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "preview_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "preview_sampler": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "ddim", "ddpm", "deis", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ipndm", "ipndm_v", "uni_pc", "uni_pc_bh2"], {"default": "euler"}),
                "preview_scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"], {"default": "normal"}),
                "preview_width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 8}),
                "preview_height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 8}),
                "preview_positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "preview_negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "preview_no_lora": ("BOOLEAN", {"default": False}),
                "preview_model": ("MODEL",),
                "preview_clip": ("CLIP",),
                "preview_vae": ("VAE",),
            },
        }

    def run(self, model, clip, model_strength, clip_strength, profile, context, vae=None, save_path="", tagging_options=None, train_options=None, preview_enable=False, preview_prompt_index=0, preview_seed=42, preview_steps=20, preview_cfg=7.0, preview_sampler="euler", preview_scheduler="normal", preview_width=512, preview_height=512, preview_positive_prompt="", preview_negative_prompt="", preview_no_lora=False, preview_model=None, preview_clip=None, preview_vae=None):
        print(f"[md_soya] === Instant Reference LoRA started ===")
        print(f"[md_soya] profile={profile}, context={'provided' if context else 'None'}, vae={'provided' if vae else 'None'}")
        print(f"[md_soya] model_strength={model_strength}, clip_strength={clip_strength}, save_path={save_path}")
        print(f"[md_soya] preview_enable={preview_enable}, preview_prompt_index={preview_prompt_index}")
        try:
            info, preview_batch = _execute_reference_lora(
                model,
                clip,
                profile,
                model_strength=model_strength,
                clip_strength=clip_strength,
                tagging_options=tagging_options,
                train_options=train_options,
                save_path=save_path,
                vae=vae,
                context=context,
                preview_enable=preview_enable,
                preview_prompt_index=preview_prompt_index,
                preview_seed=preview_seed,
                preview_steps=preview_steps,
                preview_cfg=preview_cfg,
                preview_sampler=preview_sampler,
                preview_scheduler=preview_scheduler,
                preview_width=preview_width,
                preview_height=preview_height,
                preview_positive_prompt=preview_positive_prompt,
                preview_negative_prompt=preview_negative_prompt,
                preview_no_lora=preview_no_lora,
                preview_model=preview_model,
                preview_clip=preview_clip,
                preview_vae=preview_vae,
            )
            print(f"[md_soya] === completed ===\n{info}")
            if preview_batch is None:
                preview_batch = torch.zeros((1, 64, 64, 3), device=comfy.model_management.intermediate_device())
            return (info, preview_batch,)
        except Exception as e:
            import traceback
            print(f"[md_soya] === ERROR ===\n{traceback.format_exc()}")
            raise

class TaggingOptionsV1:
    CATEGORY = "reference/training"
    RETURN_TYPES = ("TAGGING_OPTIONS",)
    RETURN_NAMES = ("tagging_options",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "general_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prepend_tags": ("STRING", {"default": "", "multiline": False}),
                "append_tags": ("STRING", {"default": "", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "replace_tags": ("STRING", {"default": "", "multiline": False}),
                "remove_underscore": ("BOOLEAN", {"default": True}),
            }
        }

    def build(self, **kwargs):
        return (kwargs,)


class TrainOptionsV1:
    CATEGORY = "reference/training"
    RETURN_TYPES = ("TRAIN_OPTIONS",)
    RETURN_NAMES = ("train_options",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps_override": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "learning_rate_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "network_dim_override": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "network_alpha_override": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "resolution_override": ("STRING", {"default": "", "multiline": False}),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "cache_latents": ("BOOLEAN", {"default": True}),
                "cache_text_encoder_outputs": ("BOOLEAN", {"default": True}),
                "train_text_encoder": ("BOOLEAN", {"default": False}),
                "text_encoder_lr": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "save_every_n_steps": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "seed_override": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "force_retrain": ("BOOLEAN", {"default": False}),
            }
        }

    def build(self, **kwargs):
        return (kwargs,)


class ContextBuilderV1:
    CATEGORY = "reference/training"
    RETURN_TYPES = ("CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "images": ("IMAGE",),
            }
        }

    def build(self, positive_prompt, negative_prompt, images):
        print(f"[md_soya] ContextBuilder: positive_prompt={repr(positive_prompt)}")
        print(f"[md_soya] ContextBuilder: negative_prompt={repr(negative_prompt)}")
        print(f"[md_soya] ContextBuilder: images shape={images.shape}")
        pos_groups = _parse_prompt_groups(positive_prompt)
        neg_groups = _parse_prompt_groups(negative_prompt)
        num_images = images.shape[0]
        print(f"[md_soya] ContextBuilder: pos_groups={pos_groups}, neg_groups={neg_groups}")

        if not pos_groups:
            raise RuntimeError(
                "Positive prompt must contain at least one [tag group]. "
                "Format: [tag1, tag2], [tag3, tag4], ..."
            )

        if len(pos_groups) != num_images:
            raise RuntimeError(
                f"Positive prompt has {len(pos_groups)} [groups] but {num_images} images. "
                f"Each [group] must match one image."
            )

        if neg_groups and len(neg_groups) != num_images:
            raise RuntimeError(
                f"Negative prompt has {len(neg_groups)} [groups] but {num_images} images. "
                f"Each [group] must match one image."
            )

        entries = []
        for i in range(num_images):
            entries.append({
                "index": i,
                "positive_tags": pos_groups[i],
                "negative_tags": neg_groups[i] if i < len(neg_groups) else "",
            })

        return ({"images": images, "entries": entries},)


NODE_CLASS_MAPPINGS = {
    "md_soya_InstantReferenceLoRA": InstantReferenceLoRAV1,
    "md_soya_ContextBuilder": ContextBuilderV1,
    "md_soya_ReferenceTaggingOptions": TaggingOptionsV1,
    "md_soya_ReferenceTrainOptions": TrainOptionsV1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "md_soya_InstantReferenceLoRA": "md_soya Instant Reference LoRA",
    "md_soya_ContextBuilder": "md_soya Context Builder",
    "md_soya_ReferenceTaggingOptions": "md_soya Reference Tagging Options",
    "md_soya_ReferenceTrainOptions": "md_soya Reference Train Options",
}
