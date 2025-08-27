
import os
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from nbclient import NotebookClient
    import nbformat
except Exception as e:
    NotebookClient = None
    nbformat = None

class NotebookExecutionError(RuntimeError):
    pass

def _detect_default_kernel() -> str:
    # Try to detect a valid kernel name from the environment
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        name = KernelSpecManager().default_kernel_name
        if isinstance(name, str) and name.strip():
            return name
    except Exception:
        pass
    # Fallbacks that usually exist
    for candidate in ("python3", "python"):
        try:
            return candidate
        except Exception:
            continue
    return "python3"

def execute_notebook(
    notebook_path: str,
    working_dir: Optional[str] = None,
    inject_context: Optional[Dict[str, Any]] = None,
    timeout: int = 1200,
    kernel_name: Optional[str] = None,
    save_output_to: Optional[str] = None
) -> None:
    """
    Execute a Jupyter notebook in-process using nbclient.
    - notebook_path: path to the .ipynb to execute
    - working_dir: cwd for execution; defaults to notebook's parent
    - inject_context: dict injected into the first cell as a Python assignment
    - timeout: per-cell execution timeout in seconds
    - kernel_name: kernel to use; if None, auto-detect a sensible default
    - save_output_to: optional path to write executed notebook for provenance
    """
    if NotebookClient is None or nbformat is None:
        raise NotebookExecutionError(
            "nbclient/nbformat not available. Please install with: pip install nbclient nbformat"
        )

    nb_path = Path(notebook_path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    nb = nbformat.read(nb_path, as_version=4)

    # Optionally inject a cell at the top with context variables
    if inject_context:
        from nbformat.v4.nbbase import new_code_cell
        import json
        payload = json.dumps(inject_context)
        runtime_name = inject_context.get("_runtime_context_name", "PIPELINE_CONTEXT")
        code = f"{runtime_name} = {payload}"
        nb.cells.insert(0, new_code_cell(source=code))

    # Resolve kernel
    resolved_kernel = kernel_name if isinstance(kernel_name, str) and kernel_name.strip() else _detect_default_kernel()

    # Configure client (avoid passing None for kernel_name)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=resolved_kernel,
        resources={"metadata": {"path": working_dir or str(nb_path.parent)}}
    )

    try:
        client.execute()
    except Exception as e:
        raise NotebookExecutionError(f"Execution failed for {notebook_path}: {e}") from e

    if save_output_to:
        out_path = Path(save_output_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_path)
