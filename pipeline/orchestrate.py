import os, sys, json, shutil, datetime
from pathlib import Path
from typing import Any, Dict
import yaml

from notebook_exec import execute_notebook, NotebookExecutionError

HERE = Path(__file__).parent

def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def make_run_artifacts(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return a dict of fully-qualified artifact paths under a timestamped run dir."""
    base = Path(cfg["artifacts"]["dir"]).resolve()
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fq = {"dir": str(run_dir)}
    for k, v in cfg["artifacts"].items():
        if k == "dir":
            continue
        fq[k] = str(run_dir / v)
    return fq

def update_latest_pointer(run_dir: Path):
    latest = run_dir.parent / "latest"
    
    # try:
    #     if latest.exists() or latest.is_symlink():
    #         if latest.is_symlink():
    #             latest.unlink()
    #         elif latest.is_dir():
    #             # remove old dir to replace with symlink
    #             shutil.rmtree(latest)
    #         else:
    #             latest.unlink()
    #     latest.symlink_to(run_dir, target_is_directory=True)
    # except Exception:
    # Fallback: copy if symlink not allowed
    if latest.exists():
        if latest.is_dir():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(run_dir, latest)

# Robust notebook path resolution
def resolve_nb_path(nb_path_str: str, cfg_path: Path) -> Path:
    """
    Resolve a notebook path that may be relative to either:
      - the pipeline/ folder (where this file and config.yaml live), or
      - the repo root (one level up from pipeline/).
    Returns an absolute Path (first found).
    """
    p = Path(nb_path_str)
    if p.is_absolute():
        return p

    pipeline_dir = cfg_path.parent            
    repo_root = pipeline_dir.parent           

    candidates = [
        (pipeline_dir / p).resolve(),         
        (repo_root / p).resolve(),            
    ]
    for c in candidates:
        if c.exists():
            return c

    # Nothing found: return the first candidate for clearer downstream error context
    return candidates[0]


def main(config_path: str = None):
    cfg_path = Path(config_path) if config_path else HERE / "config.yaml"
    cfg = load_config(cfg_path)

    # Normalize artifacts dir to an absolute path rooted at the repo root (parent of pipeline/)
    art_dir_cfg = Path(cfg["artifacts"]["dir"])
    if not art_dir_cfg.is_absolute():
        repo_root = cfg_path.parent.parent          
        art_dir_abs = (repo_root / art_dir_cfg).resolve()
        art_dir_abs.mkdir(parents=True, exist_ok=True)
        cfg["artifacts"]["dir"] = str(art_dir_abs)

    print(f"[resolve] artifacts.dir -> {cfg['artifacts']['dir']}")


    # Compute per-run artifact paths
    fq_artifacts = make_run_artifacts(cfg)

    # Shared context for notebooks
    runtime_name = cfg.get("runtime_context_name", "PIPELINE_CONTEXT")
    
    data_dir_cfg = cfg.get("data_dir", "data")
    candidates = []

    p = Path(data_dir_cfg)
    if p.is_absolute():
        candidates = [p]
    else:
        # try relative to (a) config.yaml dir (pipeline/), (b) repo root (pipeline/..)
        candidates = [
            (cfg_path.parent / p).resolve(),       
            (cfg_path.parent.parent / p).resolve(),
        ]

    data_dir_abs = next((c for c in candidates if c.exists()), candidates[-1])
    print(f"Using data_dir: {data_dir_abs}")

    params = cfg.get("params", {})
    
    context = {
        "_runtime_context_name": runtime_name,
        "params": params,
        "artifacts": fq_artifacts,
        "cwd": str(HERE),
        "data_dir": str(data_dir_abs),   # pass absolute path to notebooks
    }

    executed_dir = Path(fq_artifacts["dir"]) / "_executed_runs"
    executed_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve notebook paths robustly ---
    notebooks_cfg = cfg.get("notebooks", {})
    stages = [
        ("preprocess", notebooks_cfg.get("preprocess")),
        ("explore",    notebooks_cfg.get("explore")),
        ("train",      notebooks_cfg.get("train")),
    ]

    order = []
    for stage_name, nb_rel in stages:
        if not nb_rel:
            continue
        nb_path = resolve_nb_path(nb_rel, cfg_path)
        print(f"[resolve] {stage_name}: {nb_rel} -> {nb_path}")
        order.append((stage_name, nb_path))
    

    # Run all
    for stage, nb_path in order:
        print(f"=== Running stage: {stage} ===")
        out_ipynb = executed_dir / f"{nb_path.stem}__executed.ipynb"
        try:
            execute_notebook(
                notebook_path=str(nb_path),
                working_dir=str(nb_path.parent) if nb_path.parent.exists() else str(HERE),
                inject_context={**context, "_runtime_context_name": runtime_name},
                timeout=1800,
                kernel_name="python3",  # let nbclient resolve; change if your kernel is named differently
                save_output_to=str(out_ipynb)
            )
            print(f"[ok] {stage} completed. Executed notebook saved to: {out_ipynb}")
        except NotebookExecutionError as e:
            print(f"[error] {e}")
            sys.exit(1)

    # Update 'latest' pointer
    update_latest_pointer(Path(fq_artifacts["dir"]))

    print("Pipeline finished successfully.")
    print("Artifacts (this run):", fq_artifacts["dir"])
    print(json.dumps(fq_artifacts, indent=2))

if __name__ == "__main__":
    # Allow optional --config /path/to/config.yaml
    cfg_arg = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--config":
        cfg_arg = sys.argv[2]
    main(cfg_arg)
