from __future__ import annotations

import re
import subprocess
import traceback
from pathlib import Path


ATTENTION_MODE_PATTERN = re.compile(
    r'^(?P<prefix>\s*attn_mode\s*=\s*)"(?P<mode>[^"]+)"(?P<suffix>\s*)$',
    re.MULTILINE,
)

XFORMERS_PROBE_CODE = """
import torch
import xformers.ops as xops

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the managed training runtime")

query = torch.randn((1, 64, 8, 64), device="cuda", dtype=torch.bfloat16)
output = xops.memory_efficient_attention(query, query, query)
torch.cuda.synchronize()
if output.shape != query.shape:
    raise RuntimeError(f"unexpected xFormers output shape: {tuple(output.shape)}")
if not bool(torch.isfinite(output).all().item()):
    raise RuntimeError("xFormers output contains NaN or Inf")
"""


def _log_probe_failure(python_path: Path, detail: str) -> None:
    print(
        "[md_soya] xFormers 기능 검사 실패: "
        f"python={python_path}, detail={detail}"
    )


def xformers_is_usable(python_path: str | Path) -> bool:
    """관리형 학습 런타임에서 xFormers CUDA 연산이 실제로 동작하는지 확인한다."""
    resolved_python = Path(python_path)
    if not resolved_python.is_file():
        _log_probe_failure(resolved_python, "관리형 런타임 Python 파일이 없습니다.")
        return False

    try:
        result = subprocess.run(
            [str(resolved_python), "-c", XFORMERS_PROBE_CODE],
            cwd=str(resolved_python.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=120,
        )
    except Exception as exc:
        _log_probe_failure(
            resolved_python,
            f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()
        return False

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit_code={result.returncode}, 출력 없음"
        _log_probe_failure(resolved_python, detail[-4000:])
        return False

    print(f"[md_soya] xFormers 기능 검사 성공: python={resolved_python}")
    return True


def resolve_runtime_attention(
    config_text: str,
    python_path: str | Path,
) -> tuple[str, str]:
    """설정된 attention을 유지하거나, 사용할 수 없는 xFormers를 torch로 전환한다."""
    match = ATTENTION_MODE_PATTERN.search(config_text)
    configured_mode = match.group("mode").strip().lower() if match else ""
    if configured_mode != "xformers":
        return config_text, configured_mode

    if xformers_is_usable(python_path):
        return config_text, configured_mode

    resolved = ATTENTION_MODE_PATTERN.sub(
        lambda item: f'{item.group("prefix")}"torch"{item.group("suffix")}',
        config_text,
        count=1,
    )
    print(
        "[md_soya] attention backend fallback 적용: "
        "configured=xformers, resolved=torch, "
        "reason=관리형 학습 런타임에서 xFormers CUDA 연산을 사용할 수 없음"
    )
    return resolved, "torch"
