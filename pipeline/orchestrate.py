from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml  # requires PyYAML

# Local import
from notebook_exec import execute_notebook, NotebookExecutionError

 
# Filesystem anchors/paths
 

PIPELINE_DIR: Path = Path(__file__).resolve().parent        # .../pipeline
REPO_ROOT: Path = PIPELINE_DIR.parent                       # repo root
NB_DIR: Path = REPO_ROOT / "notebooks"                      # notebooks live at repo root
DATA_DIR: Path = REPO_ROOT / "data"                         # raw / prepared data root
ART_DIR: Path = REPO_ROOT / "artifacts"                     # pipeline outputs

 
# Utilities
 

def _assert_exists(p: Path, kind: str = "path") -> None:
    if not p.exists():
        raise FileNotFoundError(f"{kind.capitalize()} not found: {p}")

def _nb(name: str) -> Path:
    """Return absolute path to a notebook, verifying it exists."""
    p = NB_DIR / name
    _assert_exists(p, "notebook")
    return p

def _load_config(cfg_path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration. If cfg_path is None, default to pipeline/config.yaml.
    """
    cfg_file = Path(cfg_path).resolve() if cfg_path else (PIPELINE_DIR / "config.yaml")
    _assert_exists(cfg_file, "config")
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def _make_run_artifacts() -> Dict[str, str]:
    """
    Create a new run directory under artifacts/ and return key file paths.
    """
    ART_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ART_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_id": run_id,
        "dir": str(run_dir),
        # Downstream notebooks may write these (adjust if your notebooks write different names)
        "processed": str(run_dir / "processed.parquet"),
        "model": str(run_dir / "kmeans_model.joblib"),
        "summary": str(run_dir / "cluster_summary.json"),
        "selection": str(run_dir / "selection.json"),
        "elbow": str(run_dir / "elbow_plot.png"),
        "silhouette": str(run_dir / "silhouette_plot.png"),
        "db_plot": str(run_dir / "db_plot.png"),
        "ch_plot": str(run_dir / "ch_plot.png"),
    }
    return paths

def _update_latest_pointer(current: Path) -> None:
    """
    Make artifacts/latest point to the current run. Try a symlink; otherwise copy.
    """
    latest = ART_DIR / "latest"
    try:
        if latest.exists():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        latest.symlink_to(current, target_is_directory=True)
    except Exception:
        # Fallback: copy when symlinks aren't permitted
        if latest.exists():
            if latest.is_dir():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        shutil.copytree(current, latest)

 
# Main
 

def main(cfg_path: Optional[str] = None) -> str:
    """
    Execute the pipeline notebooks in order. Returns the run_id on success.
    """
    # Sanity prints (helpful in Streamlit status logs)
    print("Pipeline environment")
    print("Repo root    :", REPO_ROOT)
    print("Pipeline dir :", PIPELINE_DIR)
    print("Notebooks dir:", NB_DIR)
    print("Data dir     :", DATA_DIR)
    print("Artifacts dir:", ART_DIR)

    # Validate key dirs exist
    _assert_exists(NB_DIR, "notebooks folder")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = _load_config(cfg_path)

    # Create run artifacts directory
    artifacts = _make_run_artifacts()
    run_dir = Path(artifacts["dir"])
    print(f"\nUsing run dir: {run_dir}")

    # Prepare context injected into notebooks (accessible as __PIPELINE_CONTEXT__)
    inject_ctx = {
        "DATA_DIR": str(DATA_DIR.resolve()),
        "ART_DIR": str(ART_DIR.resolve()),
        "RUN_DIR": str(run_dir.resolve()),
        "CONFIG": cfg,
    }

    # Stage: preprocess
    print("\nRunning stage: preprocess")
    execute_notebook(
        notebook_path=str(_nb("preprocess.ipynb")),
        working_dir=str(REPO_ROOT),  # relative paths inside the notebook resolve to repo root
        inject_context=inject_ctx,
        timeout=1800,
        kernel_name=None,  # let nbclient use default (usually python3)
        save_output_to=str(run_dir / "preprocess.out.ipynb"),
    )

    # Stage: explore (optional)
    maybe_explore = NB_DIR / "explore.ipynb"
    if maybe_explore.exists():
        print("\nRunning stage: explore")
        execute_notebook(
            notebook_path=str(maybe_explore),
            working_dir=str(REPO_ROOT),
            inject_context=inject_ctx,
            timeout=1800,
            kernel_name=None,
            save_output_to=str(run_dir / "explore.out.ipynb"),
        )
    else:
        print("Skipping 'explore.ipynb' (not found).")

    # Stage: k-means / training
    print("\nRunning stage: k_means_final")
    execute_notebook(
        notebook_path=str(_nb("k_means_final.ipynb")),
        working_dir=str(REPO_ROOT),
        inject_context=inject_ctx,
        timeout=1800,
        kernel_name=None,
        save_output_to=str(run_dir / "k_means_final.out.ipynb"),
    )

    # Update 'latest' pointer
    _update_latest_pointer(run_dir)

    print("\nPipeline finished successfully.")
    print(f"Artifacts (this run): {run_dir}")
    return artifacts["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the scouting pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to pipeline/config.yaml).",
    )
    args = parser.parse_args()
    try:
        main(args.config)
    except NotebookExecutionError as e:
        # Surface a cleaner error for CI/Streamlit logs
        print(f"\nERROR: Notebook execution failed.\n{e}")
        raise
