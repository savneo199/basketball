from pathlib import Path
import os

# Resolve repo root by walking up until we find pipeline/config.yaml
def find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "pipeline" / "config.yaml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start

APP_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(APP_DIR)

DATA_DIR = ROOT / "data"
RAW_BASE = DATA_DIR / "output_by_college_clean"
ART_DIR = ROOT / "artifacts"
PIPELINE_DIR = ROOT / "pipeline"
CFG_PATH = PIPELINE_DIR / "config.yaml"

# Optional API runner (if you ever use it)
PIPELINE_API_URL = os.environ.get("PIPELINE_API_URL", "").strip()
