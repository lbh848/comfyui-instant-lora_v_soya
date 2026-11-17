"""Adapt a Comfy ConvRot checkpoint to the pinned anima_lora MOSS loader."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from library.int8.linear import base_fingerprint
from library.moss.model import MOSSImageTransformer
from library.moss.requests import variant_dir


def prepare_checkpoint(source: Path, destination: Path, digest: str, variant: str):
    transformer = variant_dir(destination, variant) / "transformer"
    transformer.mkdir(parents=True)
    with torch.device("meta"):
        model = MOSSImageTransformer()
    expected = {k: v.shape for k, v in model.state_dict().items()}
    model.save_config(transformer)
    weight_map, copies, ends = {}, {}, {}
    with safe_open(str(source), framework="pt") as weights:
        keys = set(weights.keys())
        # Store one BF16 shard per block for the backward-copy initialization.
        # Forward uses the original INT8 bytes and per-row scales, including QKV.
        for index in range(len(model.layers) + 1):
            prefix = f"layers.{index}."
            shard = {}
            names = (k for k in keys if k.startswith(prefix)) if index < len(model.layers) else (
                k for k in keys if not k.startswith("layers."))
            for name in sorted(names):
                if name.endswith((".weight_scale", ".comfy_quant", ".comfy_attention.config")):
                    continue
                tensor = weights.get_tensor(name)
                if tensor.dtype != torch.int8:
                    shard[name] = tensor
                    continue
                module = name.removesuffix(".weight")
                spec = json.loads(bytes(weights.get_tensor(module + ".comfy_quant")))
                if (spec.get("format"), spec.get("convrot"), spec.get("convrot_groupsize")) != ("int8_tensorwise", True, 256):
                    raise ValueError(f"Unsupported MOSS INT8 layout for {module}: {spec}")
                scale = weights.get_tensor(module + ".weight_scale").reshape(-1, 1)
                targets = ([module.removesuffix("qkv") + part for part in ("q", "k", "v")]
                           if module.endswith(".proj.qkv") else [module])
                for target, q, s in zip(targets, tensor.chunk(len(targets)), scale.chunk(len(targets))):
                    copies[target + ".weight"] = q
                    copies[target + ".weight_scale"] = s
                    unrotated = dequantize_int8_convrot_weight(q.cuda(), s.cuda(), 256).to(torch.bfloat16).cpu()
                    shard[target + ".weight"] = unrotated
                    if target in ("layers.0.proj.q", "layers.41.proj.ffn_down"):
                        ends[target] = SimpleNamespace(weight=unrotated)
            filename = f"diffusion_pytorch_model-{index + 1:05d}-of-{len(model.layers) + 1:05d}.safetensors"
            for name, tensor in shard.items():
                if expected.get(name) != tensor.shape:
                    raise ValueError(f"Unexpected MOSS tensor {name} with shape {tensor.shape}")
                weight_map[name] = filename
            save_file(shard, transformer / filename, metadata={"format": "pt"})
    if weight_map.keys() != expected.keys():
        raise ValueError(f"Missing MOSS tensors: {sorted(expected.keys() - weight_map.keys())}")
    index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    modules = sorted(k.removesuffix(".weight") for k in copies if k.endswith(".weight"))
    save_file(copies, destination / "forward.safetensors", metadata={
        "source": source.name, "int8_id": digest, "int8_modules": json.dumps(modules),
        "base_fingerprint": base_fingerprint([(k, ends[k]) for k in ("layers.0.proj.q", "layers.41.proj.ffn_down")]),
    })


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:3])
    prepare_checkpoint(source, destination, sys.argv[3], sys.argv[4])
