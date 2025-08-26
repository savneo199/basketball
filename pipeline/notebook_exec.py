from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

try:
    import nbformat
    from nbclient import NotebookClient
except Exception as _e:
    nbformat = None
    NotebookClient = None

class NotebookExecutionError(RuntimeError):
    """Raised when notebook execution fails."""

def _default_kernel() -> str:
    # Simple default; if you register custom kernels, adapt this.
    return "python3"

def execute_notebook(
    notebook_path: str,
    working_dir: Optional[str] = None,
    inject_context: Optional[Dict[str, Any]] = None,
    timeout: int = 1800,
    kernel_name: Optional[str] = None,
    save_output_to: Optional[str] = None,
) -> None:
    """
    Execute a .ipynb file in-process.

    Parameters
    ----------
    notebook_path : str
        Absolute or relative path to the notebook.
    working_dir : Optional[str]
        Directory to treat as the notebook's working directory (for relative I/O).
        If None, uses the notebook's parent folder.
    inject_context : Optional[Dict[str, Any]]
        If provided, injects a first cell defining __PIPELINE_CONTEXT__ = <dict>.
    timeout : int
        Cell execution timeout (seconds).
    kernel_name : Optional[str]
        Name of the Jupyter kernel to use. Defaults to "python3".
    save_output_to : Optional[str]
        If provided, write executed notebook (with outputs) to this path.
    """
    if nbformat is None or NotebookClient is None:
        raise ImportError("nbclient and nbformat are required to execute notebooks.")

    nb_path = Path(notebook_path).resolve()
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    # Load notebook
    nb = nbformat.read(nb_path, as_version=4)

    # Inject lightweight context as first cell (optional)
    if inject_context:
        import json
        ctx_src = (
            "# Auto-injected by orchestrator\n"
            f"__PIPELINE_CONTEXT__ = {json.dumps(inject_context)}\n"
        )
        nb.cells.insert(0, nbformat.v4.new_code_cell(ctx_src))

    # Kernel + execution cwd
    resolved_kernel = kernel_name or _default_kernel()
    exec_cwd = Path(working_dir).resolve() if working_dir else nb_path.parent

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=resolved_kernel,
        resources={"metadata": {"path": str(exec_cwd)}},
        allow_errors=False,
    )

    try:
        client.execute()
    except Exception as e:
        raise NotebookExecutionError(f"Execution failed for {nb_path}: {e}") from e

    if save_output_to:
        out_path = Path(save_output_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_path)
