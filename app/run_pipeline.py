import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import shutil
import streamlit as st
from get_paths import ART_DIR, PIPELINE_DIR

def run_pipeline():
    """Run pipeline/orchestrate.py locally and update artifacts/latest."""
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env["RUN_ID"] = run_id

    orch = (PIPELINE_DIR / "orchestrate.py").resolve()
    if not orch.exists():
        st.error(f"orchestrate.py not found at: {orch}")
        return None

    cmd = [sys.executable, str(orch)]
    proc = subprocess.Popen(
        cmd, cwd=str(PIPELINE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    with st.status("Running pipeline...", expanded=True) as status:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            st.write(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            status.update(label="Pipeline failed", state="error")
            st.error("Pipeline failed. See logs above.")
            return None

        latest = ART_DIR / "latest"
        current = ART_DIR / run_id
        try:
            if latest.exists():
                if latest.is_symlink() or latest.is_file():
                    latest.unlink()
                elif latest.is_dir():
                    shutil.rmtree(latest)
            latest.symlink_to(current, target_is_directory=True)
        except Exception:
            if latest.exists():
                if latest.is_dir():
                    shutil.rmtree(latest)
                else:
                    latest.unlink()
            shutil.copytree(current, latest)

        st.cache_data.clear()
        st.cache_resource.clear()
        status.update(label=f"Finished: {run_id}", state="complete")

    return run_id
