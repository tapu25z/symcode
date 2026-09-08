#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PAPER_BOOTSTRAP_PYTHON:-python3}"
venv_dir="${PAPER_VENV:-/opt/symplan-paper-v3-venv}"
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12 (3.11 recommended)"'
"$python_bin" -c 'import os, ctypes; from pathlib import Path; p=Path.cwd(); assert os.geteuid()==0 and p.stat().st_uid==0, "Use root and a root-owned checkout"; ctypes.CDLL("libseccomp.so.2")'
"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$venv_dir/bin/python" -m pip install -r requirements-paper.txt
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" -m pytest -q tests/test_paperbench.py
"$venv_dir/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unsupported"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$venv_dir/bin/python" -c 'import ctypes; ctypes.CDLL("libseccomp.so.2")'
if [[ ! -f data/paper/manifest.json ]]; then
  "$venv_dir/bin/python" -m paperbench.prepare
fi
# Process executor requires the checkout/data to be private and root owned.
"$venv_dir/bin/python" -c 'import os; from pathlib import Path; p=Path.cwd(); assert os.geteuid()==0 and p.stat().st_uid==0, "Process executor requires a root-owned checkout and root runner"; p.chmod(0o700)'
mkdir -p paper_runs
"$venv_dir/bin/python" -m pip freeze > paper_runs/server-requirements.lock.txt
