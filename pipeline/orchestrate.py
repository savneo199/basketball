
import os, sys, json, shutil, datetime
from pathlib import Path
from typing import Any, Dict
import yaml

from notebook_exec import execute_notebook, NotebookExecutionError

HERE = Path(__file__).parent

def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def _make_run_artifacts(cfg: Dict[str, Any]) -> Dict[str, str]:
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

def _update_latest_pointer(run_dir: Path):
    latest = run_dir.parent / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink():
                latest.unlink()
            elif latest.is_dir():
                # remove old dir to replace with symlink
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(run_dir, target_is_directory=True)
    except Exception:
        # Fallback: copy if symlink not allowed
        if latest.exists():
            if latest.is_dir():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        shutil.copytree(run_dir, latest)

def main(config_path: str = None):
    cfg_path = Path(config_path) if config_path else HERE / "config.yaml"
    cfg = load_config(cfg_path)

    # Compute per-run artifact paths
    fq_artifacts = _make_run_artifacts(cfg)

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
            (cfg_path.parent / p).resolve(),    # pipeline/data/...
            (cfg_path.parent.parent / p).resolve(),  # repo_root/data/...
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

    # Notebook order
    notebooks = cfg["notebooks"]
    order = [
        ("preprocess", notebooks["preprocess"]),
        ("explore",    notebooks["explore"]),
        ("train",      notebooks["train"]),
    ]

    # Run all
    for stage, nb_path in order:
        print(f"=== Running stage: {stage} ===")
        out_ipynb = executed_dir / f"{Path(nb_path).stem}__executed.ipynb"
        try:
            execute_notebook(
                notebook_path=str(nb_path),
                working_dir=str(Path(nb_path).parent) if Path(nb_path).parent.exists() else str(HERE),
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
    _update_latest_pointer(Path(fq_artifacts["dir"]))

    print("Pipeline finished successfully.")
    print("Artifacts (this run):", fq_artifacts["dir"])
    print(json.dumps(fq_artifacts, indent=2))

if __name__ == "__main__":
    # Allow optional --config /path/to/config.yaml
    cfg_arg = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--config":
        cfg_arg = sys.argv[2]
    main(cfg_arg)
