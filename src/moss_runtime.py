"""Pinned MOSS sd-scripts runtime, independent of the ANIMA/SDXL environment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
import torchvision

from .runtime import ensure_dir, ensure_uv, get_runtime_paths, resolve_runtime_python, run_command, venv_python


SD_SCRIPTS_COMMIT = "cb70b96858477e716a2096292ab8aab433277f38"
SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"


def training_vae_state_dict(state_dict):
    """Restore FLUX.2 names flattened by ComfyUI's AutoencoderKL loader."""
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("quant_conv."):
            key = "encoder." + key
        elif key.startswith("post_quant_conv."):
            key = "decoder." + key
        if key in converted:
            raise ValueError(f"Duplicate FLUX.2 VAE weight after conversion: {key}")
        converted[key] = value
    return converted


def prepare_training(profile, slots, builtins, run_dir, options, log_path, write_config):
    if options.train_text_encoder or not all((options.gradient_checkpointing,
                                              options.cache_latents, options.cache_text_encoder_outputs)):
        raise ValueError("MOSS 16GB 학습은 텍스트 인코더 학습 끄기, Gradient Checkpointing, 이미지·텍스트 캐시가 필요합니다.")
    encoder = slots["clip"].replacement
    if "w4a8" in Path(encoder).name.lower() or "gptq" in Path(encoder).name.lower():
        raise ValueError("MOSS 학습에는 qwen3.5_9b_int8_convrot.safetensors를 선택하세요. W4A8 GPTQ는 학습을 지원하지 않습니다.")

    paths = get_runtime_paths()
    root = ensure_dir(paths.root / "moss")
    engine = root / "sd-scripts"
    if not engine.exists():
        run_command(["git", "clone", "--no-checkout", SD_SCRIPTS_REPO, str(engine)], cwd=root, log_path=log_path)
    # This checkout belongs only to MOSS; the existing ANIMA/SDXL checkout is untouched.
    run_command(["git", "checkout", "--detach", SD_SCRIPTS_COMMIT], cwd=engine, log_path=log_path)
    uv = ensure_uv(paths, log_path=log_path)
    python = venv_python(root / "venv")
    if not python.is_file():
        run_command([uv, "venv", "--python", resolve_runtime_python(), "--system-site-packages", str(root / "venv")],
                    cwd=root, log_path=log_path)
    if not torch.version.cuda:
        raise RuntimeError("MOSS INT8 학습에는 CUDA PyTorch 환경이 필요합니다.")
    cuda_index = "https://download.pytorch.org/whl/cu" + torch.version.cuda.replace(".", "")
    runtime_identity = SD_SCRIPTS_COMMIT + ":" + torch.__version__ + ":" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env = {**os.environ, "PYTHONPATH": str(engine), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    # Keep compilation artifacts across runs, separate from the ANIMA backend.
    env["TORCHINDUCTOR_CACHE_DIR"] = str(ensure_dir(root / "compile_cache"))
    env["TRITON_CACHE_DIR"] = str(ensure_dir(root / "triton_cache"))
    marker = root / "venv" / ".moss_ready"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != runtime_identity:
        run_command([uv, "pip", "install", "--python", str(python), "--index-url", cuda_index,
                     "torch==" + torch.__version__, "torchvision==" + torchvision.__version__],
                    cwd=engine, log_path=log_path)
        run_command([uv, "pip", "install", "--python", str(python), "-r", str(engine / "requirements.txt"), "numpy==1.26.4"],
                    cwd=engine, log_path=log_path)
        if os.name == "nt":
            run_command([uv, "pip", "install", "--python", str(python), "triton-windows==3.6.0.post25"],
                        cwd=engine, log_path=log_path)
        run_command([str(python), str(engine / profile.script), "--help"], cwd=engine, log_path=log_path,
                    env=env)
        marker.write_text(runtime_identity, encoding="utf-8")
    config_path = write_config(profile, slots, builtins, run_dir, options, python)
    command = [str(python), "-m", "accelerate.commands.launch", "--num_processes", "1", "--num_machines", "1",
               "--num_cpu_threads_per_process", "2", "--mixed_precision", "bf16",
               str(engine / profile.script), "--config_file", str(config_path)]
    # Save separate, content-addressed model and preprocessing identities. The
    # normal node manifest and output sidecars continue to own run settings/artifacts.
    groups = {
        "model": [engine / profile.script, engine / "train_network.py", engine / "networks/lora_generic.py",
                  engine / "library/compile_utils.py",
                  *sorted((engine / "library").glob("moss_image*.py")),
                  *sorted((engine / "library").glob("convrot_int8*.py"))],
        "preprocessing": [Path(__file__), Path(__file__).with_name("nodes.py"), engine / "library/strategy_moss_image.py",
                          *sorted((engine / "library/dataset").rglob("*.py"))],
    }
    versions = {}
    for group, sources in groups.items():
        hashes = {}
        for source in sources:
            relative = source.relative_to(engine) if source.is_relative_to(engine) else Path("node") / source.name
            target = ensure_dir(run_dir / "sources" / group / relative.parent) / relative.name
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        versions[group] = {"version": f"moss-{group}-{digest[:12]}", "sha256": digest, "sources": hashes}
    data = [{"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(Path(builtins["TRAIN_DIR"]).rglob("*")) if path.is_file()]
    (run_dir / "moss-runtime.json").write_text(json.dumps({
        "sd_scripts_commit": SD_SCRIPTS_COMMIT, "command": command, "cwd": str(engine),
        "versions": versions, "dataset": data, "seed": options.seed_override if options.seed_override >= 0 else 42,
        "split": "training only; preview prompts are not a validation split",
        "config_path": str(config_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    run_command([uv, "pip", "freeze", "--python", str(python)], cwd=engine, log_path=run_dir / "packages.log", env=env)
    return config_path, (command, engine, env)
