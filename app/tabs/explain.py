import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score
from helpers.helpers import latest_artifacts, load_parquet, load_model, load_json_file, load_duckdb, BASE_MAP

def render():
    st.set_page_config(page_title="Explainability (SHAP)", layout="wide")

    # Load artifacts
    arts = latest_artifacts()
    if not arts:
        st.error("No artifacts found. Export your pipeline outputs to ART_DIR/latest first.")
        st.stop()

    required = ["rf", "feature_cols", "X_test", "y_test", "shap_values", "shap_arr", "expected_vals"]
    missing = [k for k in required if not arts.get(k) or not arts[k].exists()]
    if missing:
        st.warning(f"Missing SHAP artifacts: {', '.join(missing)}. Re-run the notebook export cell.")
        st.stop()

    rf            = load_model(arts["rf"])
    feature_cols  = load_model(arts["feature_cols"])  # list
    X_test        = load_parquet(arts["X_test"])
    y_test        = load_parquet(arts["y_test"])["cluster"].astype(int)

    # SHAP arrays
    shap_values   = load_model(arts["shap_values"])   # list or ndarray
    arr           = np.load(arts["shap_arr"])         # (n_samples, n_features, n_classes)
    expected_vals = np.load(arts["expected_vals"], allow_pickle=True)

    # Global importance precomputed
    global_imp = None
    if arts.get("global_importance") and arts["global_importance"].exists():
        gi = pd.read_csv(arts["global_importance"], index_col=0)
        gi = gi.iloc[:, 0] if gi.shape[1] == 1 else gi.squeeze()
        gi.index.name = None
        global_imp = gi.sort_values(ascending=False)

    # Rename metrics
    def pretty_names(cols: list[str]) -> list[str]:
        return [BASE_MAP.get(c, c) for c in cols]

    pretty_feature_cols = pretty_names(feature_cols)
    X_disp = X_test[feature_cols].copy()
    X_disp.columns = pretty_feature_cols

    # Archetype mapping
    def build_archetype_map() -> dict[int, str]:
        """
        Map cluster_id -> archetype name.

        Priority:
          1) cluster_summary.json -> take the insertion order of 'cluster_sizes' keys
             so that cluster 0 -> first key, cluster 1 -> second key, ...
          2) processed.duckdb majority vote (fallback)
          3) final fallback: "Cluster {k}"
        """
        # 1) From cluster_summary.json in order
        try:
            if arts.get("summary") and arts["summary"].exists():
                summ = load_json_file(arts["summary"])
                if isinstance(summ, dict) and "cluster_sizes" in summ and isinstance(summ["cluster_sizes"], dict):
                    # Python preserves JSON object insertion order
                    names_in_order = list(summ["cluster_sizes"].keys())
                    n_classes = arr.shape[2]
                    if len(names_in_order) >= n_classes:
                        return {i: str(names_in_order[i]) for i in range(n_classes)}
        except Exception:
            pass

        # 2) Fallback: processed.duckdb majority
        try:
            if arts.get("processed") and arts["processed"].exists():
                proc = load_duckdb(arts["processed"], table="processed")
                cand = [c for c in proc.columns if c.lower() == "cluster" or c.lower().endswith("_cluster")]
                if cand and "Archetype" in proc.columns:
                    cl_col = cand[0]
                    mm = (
                        proc[[cl_col, "Archetype"]]
                        .dropna()
                        .groupby(cl_col)["Archetype"]
                        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                        .to_dict()
                    )
                    out = {}
                    for k, v in mm.items():
                        try:
                            out[int(k)] = str(v)
                        except Exception:
                            pass
                    if out:
                        return out
        except Exception:
            pass

        # 3) Final fallback
        n_classes = arr.shape[2]
        return {k: f"Cluster {k}" for k in range(n_classes)}

    ARCHETYPE_MAP = build_archetype_map()

    # Build UI list from mapping in cluster-id order
    cluster_ids = sorted(np.unique(y_test))
    archetype_labels = [ARCHETYPE_MAP.get(int(k), f"Cluster {int(k)}") for k in cluster_ids]
    label_to_cluster = {label: int(k) for k, label in zip(cluster_ids, archetype_labels)}

    # Fidelity
    with st.expander("Surrogate model fidelity (vs. archetype assignment)", expanded=True):
        y_pred = rf.predict(X_test[feature_cols])
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Macro-F1", f"{f1m:.3f}")
        st.caption("How well the surrogate mimics the k-means assignments underlying the archetypes.")

    tabs = st.tabs(["Overview", "Archetypes", "Player"])

    # Overview
    with tabs[0]:
        st.subheader("Global drivers of player archetypes")
        if global_imp is None:
            shap_abs_mean = np.mean(np.abs(arr), axis=(0, 2))  # (n_features,)
            global_imp = pd.Series(shap_abs_mean, index=feature_cols).sort_values(ascending=False)
        gi_pretty = global_imp.copy()
        gi_pretty.index = pretty_names(list(gi_pretty.index))

        top_n = st.slider("Top features", 5, min(25, len(gi_pretty)), 15)
        fig = plt.figure(figsize=(7, max(4, top_n * 0.35)))
        gi_pretty.head(top_n).iloc[::-1].plot(kind="barh")
        plt.xlabel("Mean |SHAP| (avg over samples & archetypes)")
        plt.title("Global Feature Importance")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.caption("Beeswarm: distribution of feature impact across samples and archetypes.")
        shap.summary_plot(shap_values, X_disp, show=False)
        st.pyplot(plt.gcf()); plt.close(plt.gcf())

    # Archetypes
    with tabs[1]:
        st.subheader("What defines each archetype?")
        # Select by human archetype name (from JSON order)
        chosen_label = st.selectbox("Choose archetype", archetype_labels, index=0)
        k = label_to_cluster[chosen_label]

        idx_k = (y_test.values == k)
        if idx_k.any():
            imp_k = np.mean(np.abs(arr[idx_k, :, k]), axis=0)
            s_k = pd.Series(imp_k, index=feature_cols).sort_values(ascending=False)
            s_k_pretty = s_k.copy(); s_k_pretty.index = pretty_names(list(s_k_pretty.index))

            top_m = st.slider("Top features for this archetype", 5, min(20, len(s_k_pretty)), 10, key="top_archetype")
            fig = plt.figure(figsize=(7, max(4, top_m * 0.35)))
            s_k_pretty.head(top_m).iloc[::-1].plot(kind="barh")
            plt.xlabel("Mean |SHAP| within archetype")
            plt.title(f"Top Features Defining {chosen_label}")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

            st.caption("Beeswarm limited to this archetype’s samples.")
            X_sub = X_disp.loc[idx_k]
            if isinstance(shap_values, list):
                sv_sub = [sv[idx_k] for sv in shap_values]
            else:
                sv_sub = shap_values[idx_k]
            shap.summary_plot(sv_sub, X_sub, show=False)
            st.pyplot(plt.gcf()); plt.close(plt.gcf())
        else:
            st.info("No samples for this archetype in the current test split.")

    # Player
    with tabs[2]:
        st.subheader("Why is this player in that archetype?")

        # Prefer "Player Name - College - Season"
        label_series = None
        try:
            if arts.get("processed") and arts["processed"].exists():
                meta_df = load_duckdb(arts["processed"], table="processed")
                common_idx = meta_df.index.intersection(X_test.index)
                meta_sub = meta_df.loc[common_idx]

                name_col    = next((c for c in meta_sub.columns if c.lower() in {"player_ind", "player", "player_name"}), None)
                college_col = next((c for c in meta_sub.columns if c.lower() == "college"), None)
                season_col  = next((c for c in meta_sub.columns if "season" in c.lower()), None)

                if name_col and college_col and season_col and len(meta_sub):
                    labels = meta_sub.apply(lambda r: f"{r[name_col]} - {r[college_col]} - {r[season_col]}", axis=1)
                    label_series = pd.Series(labels.values, index=meta_sub.index)
                    X_for_pick = X_test.loc[label_series.index]
                    y_for_pick = y_test.loc[label_series.index]
                else:
                    label_series = None
        except Exception:
            label_series = None

        if label_series is not None and len(label_series):
            selection = st.selectbox("Pick a player", label_series.values)
            pos = np.where(label_series.values == selection)[0][0]
            example_idx = label_series.index[pos]
            X_src = X_for_pick
            y_src = y_for_pick
        else:
            st.info("Could not find (Player Name - College - Season) metadata; showing row index picker.")
            example_idx = st.selectbox("Pick a row index", list(X_test.index), index=0)
            X_src = X_test
            y_src = y_test

        row_pos = X_src.index.get_loc(example_idx)
        pred_class = int(rf.predict(X_src.loc[[example_idx], feature_cols])[0])
        pred_label = ARCHETYPE_MAP.get(pred_class, f"Cluster {pred_class}")
        st.write(f"**Predicted archetype:** {pred_label}")

        # Single-row SHAP vector for predicted class
        sv_row = arr[row_pos, :, pred_class]
        feat_vals = X_src.loc[example_idx, feature_cols].values

        # Expected value resolution
        ev = np.array(expected_vals)
        if ev.ndim == 0:
            base_val = float(ev)
        elif ev.ndim == 1 and ev.size > pred_class:
            base_val = float(ev[pred_class])
        else:
            base_val = float(np.ravel(ev)[0])

        # Waterfall plot (pretty names)
        try:
            fig = plt.figure()
            shap.plots._waterfall.waterfall_legacy(
                base_value=base_val,
                shap_values=sv_row,
                feature_names=pretty_feature_cols,
                features=feat_vals,
                max_display=15,
                show=False
            )
            st.pyplot(fig); plt.close(fig)
        except Exception:
            exp = shap.Explanation(
                values=sv_row,
                base_values=base_val,
                data=feat_vals,
                feature_names=pretty_feature_cols
            )
            fig = plt.figure()
            shap.plots.waterfall(exp, max_display=15, show=False)
            st.pyplot(fig); plt.close(fig)

        # Top pushes for this player
        contrib = pd.Series(sv_row, index=pretty_feature_cols).sort_values(key=np.abs, ascending=False)
        st.caption("Strongest pushes (by |SHAP|) for this player:")
        st.dataframe(contrib.head(15).to_frame("SHAP").style.format("{:.3f}"), use_container_width=True)

    # Footer
    st.caption("SHAP explains the surrogate Random Forest trained to mimic the k-means assignments (archetypes). Attributions reflect correlations, not causality.")
