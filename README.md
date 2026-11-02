# ComfyUI Instant Reference

A custom node for ComfyUI that turns a reference image set into a quick LoRA and applies it back to the current workflow.

It is aimed at fast character or style adaptation runs with minimal setup. The node prepares captions, runs a lightweight training profile, caches matching runs, and reuses the generated LoRA when possible.
Under the hood, it uses `sd-scripts` for tagging and LoRA training.

This node is still fairly rough and has only been tested with simple workflows so far. If you run into issues, please open an issue with your workflow details and logs.

## Example

![Reference adaptation example](./assets/readme-example.png)

## Workflow Example

![Workflow example](./assets/workflows.png)
Drop `assets/workflows.png` into ComfyUI to load the included SDXL example workflow.

## Train/Apply Split Workflow Example

![Train/apply workflow example](./assets/workflow-train-apply.png)
Example layout using `Instant Reference LoRA Train` followed by `Instant Reference LoRA Apply`, with the trained `lora_stack` passed into the apply node.

## Nodes

### Instant Reference LoRA

Main combined node. It takes a reference `IMAGE` batch plus whatever typed inputs the selected profile declares with `{{name:TYPE}}` tokens, prepares captions, runs the selected training profile, caches identical runs, and outputs the patched `MODEL`, patched `CLIP`, generated `lora_path`, and `lora_stack`.

### Instant Reference LoRA Train

Training-only node. It uses the same profile-driven inputs and cache behavior as the combined node, but only trains or reuses a LoRA and outputs it as `lora_stack` for downstream nodes.

### Instant Reference LoRA Apply

Apply-only node. It takes an incoming `lora_stack`, applies each LoRA entry in order to the provided `MODEL` and `CLIP`, and passes the stack through unchanged.

### Reference Tagging Options

Helper node for caption generation settings. Use it to control WD tagger thresholds and basic caption cleanup such as prepending tags, appending tags, excluding tags, replacing tags, and underscore removal.

### Reference Train Options

Helper node for training overrides. Use it to override steps, learning rate, network size, alpha, resolution, seed, caching behavior, and whether to force retraining instead of reusing a cached result.

## Profiles

### SDXL Reference LoRA

Default SDXL-oriented profile based on `sdxl_train_network.py`. It trains a LoCon-style LoRA at `1024x1024`, uses `bf16`, keeps the run short with `50` default steps, and is meant for fast reference adaptation on SDXL checkpoints. Its profile declares only `{{model:MODEL}}`, so only a model slot is exposed alongside the images.

### Anima Reference LoRA

An Anima-oriented profile based on `anima_train_network.py`. It also runs at `1024x1024` with `50` default steps, trains a lightweight LoRA for the UNet only, and its profile declares `{{model:MODEL}}`, `{{clip:CLIP}}`, and `{{vae:VAE}}`, so those sockets are exposed automatically.

### Anima LoRA V2

`md_soya Anima LoRA V2` and `md_soya Anima Train Options V2` use
[anima_lora](https://github.com/sorryhyun/anima_lora) through its public preprocessing
and training commands. The existing nodes and their sd-scripts backend remain available.
The character and style examples are in `workflows/*(anima)_v2.json`.

V2 uses the Python executable running ComfyUI. It enables latent/text embedding caches,
the engine's fused attention projections, per-block compilation and activation memory
budgeting. `use_timestep_mask` enables optional T-LoRA. Attention defaults to PyTorch
SDPA; selecting `flash` requires a compatible FlashAttention installation. `RESOLUTION`
matches V1: `1024` means a maximum area of 1024 x 1024; `width,height` uses their product.
Preprocessing keeps smaller images at their native size, reduces larger images while
preserving aspect ratio, and center-crops to V1's bucket grid (64 pixels by default).
Each execution starts a new training run. Preview, output handling and checkpoint
retention reuse the existing node pipeline.

V2 generated previews apply the registered `DiTSpectrumPatch` node before the existing
sampler when `preview_spectrum` is enabled (default). This also applies to the optional
no-LoRA comparison preview. The patch adds
Spectrum only; modulation guidance, SMC-CFG, DCW and SPD are not enabled. Install and
load `comfyui-spectrum-ksampler` to generate V2 previews. Existing V1 previews retain
their sampling behavior. Instance previews still reuse the input image.

Preview sampler and scheduler choices follow the installed ComfyUI registries, including
`er_sde`. In a local fixed-seed comparison, Spectrum with `euler_ancestral` / `normal`
produced grainy previews even without a LoRA; disabling Spectrum or using `euler` /
`normal` removed that artifact. The V2 examples use the verified Spectrum combination
`er_sde` / `simple`, 26 steps and CFG 5. Check previews separately from training.
V2 disables TensorBoard logs and progress JSONL output; the node retains its diagnostic
`run.log`. The resolved image dimensions are printed during preprocessing and recorded
in the run's `commands.json` for comparisons.

The local installation uses these source checkouts under this node's `runtime/`:

- `anima_lora`: commit `eca845a5e059208509b4f9a922a7c2ad249b10bf`.
- `anime_tools`: tag `v0.9.3`, commit `1fac04b4eb4b503bdac9ebd7a6ad6b5261e6f4da`.

The subprocess imports those sources directly. Upstream package metadata requires
Python 3.13, but the ANIMA training path was smoke-tested locally with the existing
ComfyUI Python 3.12.11, PyTorch 2.10.0+cu130 and triton-windows 3.6.0.post25. Both
ordinary LoRA and T-LoRA completed three steps and saved checkpoints on an RTX 4080.
This does not establish compatibility of the upstream GUI or other model families.

Additional packages installed into the existing ComfyUI environment were
`voluptuous`, `tomlkit`, `imagesize`, `tensorboard`, `prodigyopt`,
`pytorch-optimizer` and `softtorch`, with their dependencies. No new virtual
environment is created by the V2 nodes. Engine sources and packages must be installed
before execution; `engine_path` can select an existing anima_lora checkout.
