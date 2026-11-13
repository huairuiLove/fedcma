#!/bin/bash
# Linux counterpart of run_local.ps1. Only the requested three methods.
set -e
PY=${PY:-python}

"$PY" - <<'PY'
import json, subprocess, sys
from datetime import datetime
from pathlib import Path
root = Path.cwd()
settings = json.loads((root / 'configs/local_ihm.json').read_text())
command = [sys.executable, '-u', 'run_experiments.py', '--out_dir',
           str(root / 'experiments' / datetime.now().strftime('local_%Y%m%d_%H%M%S'))]
for key, value in settings.items():
    command += ['--' + key] + [str(x) for x in (value if isinstance(value, list) else [value])]
subprocess.run(command, check=True)
PY
