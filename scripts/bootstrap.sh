#!/usr/bin/env bash
# Install into this project only; no sudo and no shell-profile modifications.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p .tools
if [[ ! -x .tools/uv ]]; then
  python3 - <<'PY'
import io
import platform
import tarfile
import urllib.request
from pathlib import Path

if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("This bootstrap targets Linux x86_64; use README setup on other platforms")
version = "0.13.5"
bundle = "uv-x86_64-unknown-linux-gnu"
url = f"https://github.com/astral-sh/uv/releases/download/{version}/{bundle}.tar.gz"
print(f"Downloading uv {version}", flush=True)
with urllib.request.urlopen(url, timeout=120) as response:
    archive = response.read()
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
    binary = tar.extractfile(f"{bundle}/uv")
    if binary is None:
        raise SystemExit("uv archive contains no expected executable")
    target = Path(".tools/uv")
    target.write_bytes(binary.read())
    target.chmod(0o755)
PY
fi
if [[ ! -d .venv ]]; then
  .tools/uv venv --python 3.12 .venv
fi
.tools/uv pip install --python .venv/bin/python torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
.tools/uv pip install --python .venv/bin/python -r requirements-experiments.txt
.tools/uv pip freeze --python .venv/bin/python > environment.freeze.txt
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
