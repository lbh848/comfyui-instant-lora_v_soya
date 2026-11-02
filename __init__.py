import sys
import os
import runpy
import traceback

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from src.extension import comfy_entrypoint
from src.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import src.server as _server

try:
    runpy.run_path(os.path.join(_pkg_dir, "install.py"), run_name="__main__")
except Exception:
    print("[Anima V2] 자동 준비에 실패했습니다. V2 실행 시 다시 시도하며 V1은 계속 사용할 수 있습니다.")
    traceback.print_exc()

WEB_DIRECTORY = "./web/js"

__all__ = [
    "WEB_DIRECTORY",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "comfy_entrypoint",
]
