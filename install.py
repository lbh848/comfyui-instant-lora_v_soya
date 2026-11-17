"""Prepare the V2 backend during node installation or ComfyUI node loading."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

from filelock import FileLock
from packaging.requirements import Requirement


def source_ready(destination: Path, revision: str, entry: str) -> bool:
    if not (destination / entry).is_file():
        return False
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=destination,
        text=True, capture_output=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return completed.returncode == 0 and completed.stdout.strip() == revision


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    runtime = root / "runtime"
    requirements = root / "requirements.txt"
    sources = (
        ("anima_lora", "https://github.com/sorryhyun/anima_lora.git",
         "c52491c6c5664e5a6aa9f0defafd6255a0410548", "train.py"),
        ("anime_tools", "https://github.com/sorryhyun/anime_tools.git",
         "f5e0c69fd1d7554cef566ea5aafc9b0ee18bdf67", "anime_tools/__init__.py"),
    )
    if sys.version_info < (3, 12):
        print("[Anima V2] Python 3.12 이상이 필요하여 V2 준비를 생략합니다. V1은 계속 사용할 수 있습니다.")
    else:
        try:
            runtime.mkdir(parents=True, exist_ok=True)
            # Several managed Comfy instances can load this same node at once.
            with FileLock(str(runtime / ".anima_v2_setup.lock"), timeout=900):
                missing = []
                for line in requirements.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    requirement = Requirement(line)
                    if requirement.marker and not requirement.marker.evaluate():
                        continue
                    try:
                        installed = importlib.metadata.version(requirement.name)
                    except importlib.metadata.PackageNotFoundError:
                        missing.append(requirement.name)
                        continue
                    if installed not in requirement.specifier:
                        missing.append(str(requirement))

                signature = hashlib.sha256(
                    Path(__file__).read_bytes() + requirements.read_bytes()
                    + str(Path(sys.executable).resolve()).encode("utf-8")
                    + sys.version.encode("utf-8")
                    + importlib.metadata.version("torch").encode("utf-8")
                ).hexdigest()
                marker = runtime / ".anima_v2_ready"
                source_readiness = {name: source_ready(runtime / name, revision, entry)
                                    for name, _, revision, entry in sources}
                ready = (
                    not missing
                    and all(source_readiness.values())
                    and marker.is_file()
                    and marker.read_text(encoding="utf-8").strip() == signature
                )
                if not ready:
                    print("[Anima V2] 학습 엔진 자동 준비를 시작합니다.", flush=True)
                    env = os.environ.copy()
                    env.pop("PYTHONHOME", None)
                    env["PYTHONUTF8"] = "1"
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["PYTHONNOUSERSITE"] = "1"
                    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    if missing:
                        uv = shutil.which("uv")
                        if uv is None:
                            raise RuntimeError("V2 패키지 자동 설치에 필요한 uv 실행 파일을 찾을 수 없습니다.")
                        print(f"[Anima V2] 패키지 설치: {', '.join(missing)}", flush=True)
                        subprocess.run(
                            [uv, "pip", "install", "--python", sys.executable, "-r", str(requirements)],
                            cwd=root, env=env, check=True, timeout=900, creationflags=creationflags,
                        )

                    for name, repository, revision, entry in sources:
                        destination = runtime / name
                        if source_readiness[name]:
                            print(f"[Anima V2] 기존 엔진 재사용: {name}", flush=True)
                            continue
                        print(f"[Anima V2] 엔진 다운로드: {name}", flush=True)
                        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=runtime) as temporary:
                            staging = Path(temporary) / name
                            staging.mkdir()
                            for command in (
                                ["git", "init"],
                                ["git", "remote", "add", "origin", repository],
                                ["git", "fetch", "--depth", "1", "origin", revision],
                                ["git", "checkout", "--detach", "FETCH_HEAD"],
                            ):
                                subprocess.run(
                                    command, cwd=staging, env=env, check=True,
                                    timeout=600, creationflags=creationflags,
                                )
                            if destination.exists():
                                if not destination.resolve().is_relative_to(runtime.resolve()):
                                    raise RuntimeError(f"Backend path is outside the runtime directory: {destination}")
                                backup_root = runtime / "backups"
                                backup_root.mkdir(exist_ok=True)
                                backup = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=backup_root)) / name
                                shutil.move(str(destination), str(backup))
                                print(f"[Anima V2] 기존 엔진 백업: {backup}", flush=True)
                            staging.replace(destination)

                    engine = runtime / "anima_lora"
                    env["PYTHONPATH"] = os.pathsep.join((str(engine), str(runtime / "anime_tools")))
                    # Exercise the same public entry points used by the V2 node.
                    for entry in ("train.py", "scripts/preprocess/cache_latents.py",
                                  "scripts/preprocess/cache_text_embeddings.py"):
                        print(f"[Anima V2] 학습 실행 준비 확인: {entry}", flush=True)
                        completed = subprocess.run(
                            [sys.executable, entry, "--help"], cwd=engine, env=env,
                            text=True, encoding="utf-8", errors="replace", capture_output=True,
                            timeout=180, creationflags=creationflags,
                        )
                        if completed.returncode:
                            print(completed.stdout, completed.stderr, flush=True)
                            completed.check_returncode()
                    marker.write_text(signature + "\n", encoding="utf-8")
                    print("[Anima V2] 학습 엔진 자동 준비 완료", flush=True)
        except Exception:
            print(f"[Anima V2] 자동 준비 실패: python={sys.executable}, runtime={runtime}", flush=True)
            traceback.print_exc()
            raise
