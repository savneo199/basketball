
# End-to-end Pipeline (preprocess → explore → k_means_final)

This scaffold executes your three notebooks in order and saves artifacts for a dashboard.

## Layout
- `config.yaml` — paths & parameters
- `orchestrate.py` — runs the notebooks with a shared runtime context
- `train.py` — CLI wrapper
- `utils/notebook_exec.py` — notebook execution helper (uses `nbclient`)
- `artifacts/` — outputs written by your notebooks
- `dashboard_streamlit.py` — minimal app that loads artifacts

## How it works
We inject a dict called `PIPELINE_CONTEXT` (name configurable) into each notebook before execution:
```python
PIPELINE_CONTEXT = {
  "params": {...},
  "artifacts": {...},  # resolved file paths
  "cwd": "..."
}
```
In your notebooks, read it like:
```python
ctx = PIPELINE_CONTEXT
params = ctx["params"]
paths = ctx["artifacts"]
```
Then **write outputs** to the artifact paths, for example in `k_means_final.ipynb`:
```python
import joblib, json
joblib.dump(kmeans, paths["dir"] + "/kmeans_model.joblib")
joblib.dump(scaler, paths["dir"] + "/scaler.joblib")
joblib.dump(pca, paths["dir"] + "/pca.joblib")
json.dump(cluster_summary, open(paths["dir"] + "/cluster_summary.json","w"))
```

## Running the pipeline
1. Make sure dependencies are installed:
   ```bash
   pip install nbclient nbformat joblib pandas pyarrow scikit-learn pyyaml plotly streamlit
   ```
2. Run:
   ```bash
   python train.py
   # or specify a different config:
   python train.py --config /path/to/config.yaml
   ```
3. Start the dashboard:
   ```bash
   streamlit run dashboard_streamlit.py
   ```

## Notes
- The **explore** notebook usually generates EDA visuals; it can also compute metrics
  (elbow, silhouette) and save them to artifact files configured in `config.yaml`.
- If your final notebook *auto-selects* `n_clusters`, still write the chosen value into
  the summary JSON so the dashboard can display it.
- If some magics (`%`) or shell `!` commands are used, `nbclient` will generally handle
  them because they run within a kernel. If you run into errors, prefer replacing shell
  calls with Python equivalents.
