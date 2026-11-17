"""Run the pinned MOSS trainer with reusable backward weights for a frozen INT8 base."""
from __future__ import annotations

import hashlib
import os
from functools import partial
from pathlib import Path
import tempfile
import time


def backward_cache_directory(forward_file: Path) -> Path:
    import torch
    from comfy_kitchen.backends.eager import quantization
    from library.int8 import convrot, linear
    from library.moss.dit_int8 import base
    from importlib.metadata import version

    # The parent is already named after the SHA-256 of the original checkpoint.
    recipe = hashlib.sha256()
    for source in (__file__, base.__file__, linear.__file__, convrot.__file__, quantization.__file__):
        recipe.update(Path(source).read_bytes())
    recipe.update(str((torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(),
                       version("comfy-kitchen"))).encode())
    return forward_file.parent / f"backward-{recipe.hexdigest()[:20]}"


def restore_block(block, index, copies, backward):
    import torch
    from library.int8.linear import HeldCopy
    from library.moss.dit_int8 import base

    # Keep the original q/k/v module objects: the new character's LoRA already
    # references them. The frozen base runs a single fused projection upstream.
    parts = [block.proj[name] for name in base.QKV]
    block.qkv = torch.nn.Linear(parts[0].in_features, sum(p.out_features for p in parts),
                                bias=False, device="meta")
    for part in parts:
        del part.weight
        base._empty_weight(part)
        part.__class__ = base.QKVPart

    for name in ("qkv", *base.REST):
        if name == "qkv":
            module = block.qkv
            rows = [copies[f"layers.{index}.proj.{part}"] for part in base.QKV]
            forward = tuple(torch.cat([tensor.cuda() for tensor in group]) for group in zip(*rows))
        else:
            module = block.proj[name]
            forward = copies[f"layers.{index}.proj.{name}"]
        # The forward copy supplies the actual weights. Discard the BF16
        # parameter before the native converter moves the module onto CUDA.
        module.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
        base.convert_moss_(module, "infer", forward)
        module.held_wq_t = HeldCopy(backward.get_tensor(name + ".weight").cuda())
        module.register_buffer("s_t", backward.get_tensor(name + ".scale").cuda(), persistent=False)
        module.int8_mode = "full"


def convert_blocks_with_cache(transformer, mode, copies, *, directory, convert_blocks):
    from filelock import FileLock
    from library.moss.dit_int8 import base
    from safetensors import SafetensorError, safe_open
    from safetensors.torch import save_file

    if mode != "full" or not copies:
        return convert_blocks(transformer, mode, copies)
    started = time.monotonic()
    with FileLock(str(directory) + ".lock"):
        if directory.is_dir():
            for index, block in enumerate(transformer.layers):
                with safe_open(directory / f"{index}.safetensors", framework="pt") as backward:
                    restore_block(block, index, copies, backward)
            print(f"[MOSS V2] Restored frozen INT8 blocks in {time.monotonic() - started:.1f}s", flush=True)
            return

        convert_blocks(transformer, mode, copies)
        # Publish all blocks together. Only frozen tensors are saved, before
        # the native stream takes ownership of the backward buffers.
        try:
            with tempfile.TemporaryDirectory(dir=directory.parent) as stage:
                candidate = Path(stage) / "backward"
                candidate.mkdir()
                for index, block in enumerate(transformer.layers):
                    tensors = {}
                    for name in ("qkv", *base.REST):
                        module = block.qkv if name == "qkv" else block.proj[name]
                        tensors[name + ".weight"] = module.wq_t.cpu()
                        tensors[name + ".scale"] = module.s_t.cpu()
                    save_file(tensors, candidate / f"{index}.safetensors")
                candidate.rename(directory)
            print(f"[MOSS V2] Cached frozen INT8 blocks in {time.monotonic() - started:.1f}s", flush=True)
        except (OSError, SafetensorError) as exc:
            print(f"[MOSS V2] Could not store the backward cache: {exc}. Continuing training.", flush=True)


def main():
    from library.moss.requests import TrainRequest

    request = TrainRequest.from_argv()
    if request.expandable_segments:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from library.moss.dit_int8 import base
    from library.moss.train import run_train

    if request.int8 != "stream" or not request.int8_file or request.int8_file == "rtn":
        return run_train(request)
    # The upstream trainer has no conversion callback. Adapt this one step in
    # the dedicated child process; the normal training loop owns each run.
    directory = backward_cache_directory(Path(request.int8_file))
    convert_blocks = base.convert_blocks_
    base.convert_blocks_ = partial(convert_blocks_with_cache, directory=directory, convert_blocks=convert_blocks)
    try:
        return run_train(request)
    finally:
        base.convert_blocks_ = convert_blocks


if __name__ == "__main__":
    main()
