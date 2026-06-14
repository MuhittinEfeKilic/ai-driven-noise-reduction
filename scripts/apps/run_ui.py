from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    app_path = project_root / "src" / "web" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    raise SystemExit(subprocess.call(cmd, cwd=str(project_root)))
