"""bf16 미지원 GPU 자동 fp16 강등 로직 검증 스크립트.

직접 실행: comfy/.venv 파이썬으로 실행한다.
  comfy/.venv/Scripts/python.exe tests/verify_precision_fallback.py

실제 2070 하드웨어가 없어도 torch.cuda.get_device_capability 를 모킹해서
강등 분기가 올바로 동작하는지 확인한다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest import mock

# ComfyUI 루트(comfy/, comfy_api/, folder_paths.py) + 노드 루트(src 패키지)
_HERE = Path(__file__).resolve().parent
_NODE_ROOT = _HERE.parent
_COMFY_ROOT = _NODE_ROOT.parents[1]
sys.path.insert(0, str(_COMFY_ROOT))
sys.path.insert(0, str(_NODE_ROOT))

import torch  # noqa: E402
from src import nodes  # noqa: E402


def _mock_gpu(capability):
    return (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=0),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=capability),
    )


def _read_key(text: str, key: str) -> str | None:
    match = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


_results: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    _results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def gpu_supports(capability):
    patches = _mock_gpu(capability)
    with patches[0], patches[1], patches[2]:
        return nodes._gpu_supports_bf16()


def gpu_supports_no_cuda():
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        return nodes._gpu_supports_bf16()


def apply_options(cfg: str, capability):
    patches = _mock_gpu(capability)
    with patches[0], patches[1], patches[2]:
        out = nodes._apply_train_options(cfg, nodes.TrainOptions())
    return out


BF16_CFG = (
    'mixed_precision = "bf16"\n'
    'save_precision = "bf16"\n'
    'resolution = "1024,1024"\n'
)
FP16_CFG = (
    'mixed_precision = "fp16"\n'
    'save_precision = "fp16"\n'
)
NO_PRECISION_CFG = 'resolution = "1024,1024"\n'


def main() -> int:
    print("== _gpu_supports_bf16 ==")
    check("2070 (7,5) -> 미지원(False)", gpu_supports((7, 5)) is False)
    check("Ampere (8,6) -> 지원(True)", gpu_supports((8, 6)) is True)
    check("Ada/L4 (8,9) -> 지원(True)", gpu_supports((8, 9)) is True)
    check("CUDA 없음 -> 기본값(True)", gpu_supports_no_cuda() is True)

    print("\n== _apply_train_options 강등 ==")

    out = apply_options(BF16_CFG, (7, 5))
    check("bf16 cfg + 2070 -> mixed_precision fp16", _read_key(out, "mixed_precision") == "fp16")
    check("bf16 cfg + 2070 -> save_precision fp16", _read_key(out, "save_precision") == "fp16")

    out = apply_options(BF16_CFG, (8, 9))
    check("bf16 cfg + Ada -> mixed_precision bf16 유지", _read_key(out, "mixed_precision") == "bf16")
    check("bf16 cfg + Ada -> save_precision bf16 유지", _read_key(out, "save_precision") == "bf16")

    out = apply_options(FP16_CFG, (7, 5))
    check("fp16 cfg + 2070 -> fp16 유지(강등 안 함)", _read_key(out, "mixed_precision") == "fp16")

    out = apply_options(NO_PRECISION_CFG, (7, 5))
    check("precision 없는 cfg + 2070 -> 강등 미추가", _read_key(out, "mixed_precision") is None)

    # 현재 머신 실제 GPU 기준 (bf16 지원이면 강등이 발생하면 안 됨)
    real_bf16 = nodes._gpu_supports_bf16()
    print(f"\n[현재 머신] _gpu_supports_bf16() = {real_bf16}")
    out = nodes._apply_train_options(BF16_CFG, nodes.TrainOptions())
    if real_bf16:
        check("현재 머신 bf16 지원 -> config 그대로 유지", _read_key(out, "mixed_precision") == "bf16")
    else:
        check("현재 머신 bf16 미지원 -> fp16 강등", _read_key(out, "mixed_precision") == "fp16")

    print("\n== 실제 anima 프로파일 config 종단 간(2070 모킹) ==")
    import tomllib
    import traceback
    anima_path = _NODE_ROOT / "profiles" / "anima.toml"
    try:
        with anima_path.open("rb") as f:
            anima_cfg = tomllib.load(f)["config"]
        rendered = apply_options(anima_cfg, (7, 5))
        tmp = _HERE / "_tmp_anima.toml"
        tmp.write_text(rendered, encoding="utf-8")
        accel_mp = nodes._accelerate_mixed_precision(tmp)
        tmp.unlink(missing_ok=True)
        print(f"accelerate 에 전달될 --mixed_precision = {accel_mp!r}")
        for line in rendered.splitlines():
            if any(k in line for k in ("mixed_precision", "save_precision", "attn_mode")):
                print("   ", line)
        check("anima 2070 -> accelerate 가 fp16 전달", accel_mp == "fp16")
        check("anima 2070 -> save_precision fp16", _read_key(rendered, "save_precision") == "fp16")
    except Exception as exc:
        print(f"  anima config 확인 실패: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    failed = [name for name, ok in _results if not ok]
    total = len(_results)
    print(f"\n{total - len(failed)}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
