import io, zipfile, shutil
import pandas as pd
import streamlit as st

from get_paths import RAW_BASE, ART_DIR
from helpers.helpers import (
    norm_key, slugify_columns, read_slug_csv, validate_pair, summarise_saved,
    extract_zip_file, COLLEGE_MAP,
)

def render():
    st.subheader("Upload new team-season CSVs and classify archetypes")

    mode = st.radio(
        "Upload mode",
        (
            "Single season of one college",
            "Multiple seasons of one college (ZIP)",
            "Multiple seasons of multiple colleges (ZIP)",
        ),
        index=0,
    )

    st.markdown("**Each season needs two CSVs:** `individual_stats_overall.csv` and `team_stats.csv`")

    # MODE 1
    if mode == "Single season of one college":
        col_t1, col_t2 = st.columns([2, 1])
        team_display_in = col_t1.text_input("Team (display name)", placeholder="e.g., Fairfield Stags")
        season_in = col_t2.text_input("Season (folder name)", placeholder="e.g., 2024-25")

        up_ind = st.file_uploader("Upload individual_stats_overall.csv", type=["csv"], key="up_ind_single")
        up_team = st.file_uploader("Upload team_stats.csv", type=["csv"], key="up_team_single")

        if st.button("Validate & Save"):
            if not team_display_in or not season_in or not up_ind or not up_team:
                st.error("Please provide team, season, and both CSV files.")
                return
            team_key = norm_key(team_display_in)
            try:
                df_ind = slugify_columns(pd.read_csv(up_ind))
                df_team = slugify_columns(pd.read_csv(up_team))
            except Exception as e:
                st.error(f"Failed to read CSVs: {e}")
                return

            miss_ind, miss_team = validate_pair(df_ind, df_team)
            if miss_ind or miss_team:
                st.error("Validation failed. Please fix the following:")
                if miss_ind:
                    st.write("**individual_stats_overall.csv**")
                    st.code("\n".join(miss_ind))
                if miss_team:
                    st.write("**team_stats.csv**")
                    st.code("\n".join(miss_team))
                return

            dest_dir = RAW_BASE / team_key / season_in
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "individual_stats_overall.csv").write_bytes(up_ind.getvalue())
            (dest_dir / "team_stats.csv").write_bytes(up_team.getvalue())
            st.success(f"Saved files to: {dest_dir}")

            with st.expander("Preview (first 5 rows, individual)"):
                st.dataframe(df_ind.head(), use_container_width=True)
            with st.expander("Preview (first 5 rows, team)"):
                st.dataframe(df_team.head(), use_container_width=True)

    # MODE 2 
    elif mode == "Multiple seasons of one college (ZIP)":
        team_display_in = st.text_input("Team (display name)", placeholder="e.g., Fairfield Stags")
        st.caption("Upload a ZIP with structure: `college/season/{individual_stats_overall.csv, team_stats.csv}`.")
        up_zip = st.file_uploader("Upload ZIP", type=["zip"], key="up_zip_one_college")

        if st.button("Validate & Save"):
            if not team_display_in or not up_zip:
                st.error("Please provide team and a ZIP file.")
                return

            team_key = norm_key(team_display_in)
            dest_root = RAW_BASE / team_key
            dest_root.mkdir(parents=True, exist_ok=True)

            try:
                zf = zipfile.ZipFile(io.BytesIO(up_zip.getvalue()))
            except Exception as e:
                st.error(f"Could not open ZIP: {e}")
                return

            temp_extract = ART_DIR / "_tmp_extract_one_college"
            if temp_extract.exists():
                shutil.rmtree(temp_extract)
            temp_extract.mkdir(parents=True, exist_ok=True)

            extracted = extract_zip_file(zf, temp_extract)

            seasons_found = {}
            for p in extracted:
                rel = p.relative_to(temp_extract)
                parts = rel.parts
                if len(parts) < 3:
                    continue
                # Strict: college/season/file.csv (ignore college name here)
                _, season, fname = parts[0], parts[1], parts[-1]
                if fname.lower() == "individual_stats_overall.csv":
                    seasons_found.setdefault(season, {})["ind"] = p
                elif fname.lower() == "team_stats.csv":
                    seasons_found.setdefault(season, {})["team"] = p

            saved_rows, errors = [], []
            for season, files in sorted(seasons_found.items()):
                ind_p, team_p = files.get("ind"), files.get("team")
                if not ind_p or not team_p:
                    errors.append(f"{season}: missing required CSV(s).")
                    continue
                try:
                    df_ind = read_slug_csv(ind_p)
                    df_team = read_slug_csv(team_p)
                    miss_ind, miss_team = validate_pair(df_ind, df_team)
                    if miss_ind or miss_team:
                        msg = []
                        if miss_ind: msg.append("individual_stats_overall.csv: " + "; ".join(miss_ind))
                        if miss_team: msg.append("team_stats.csv: " + "; ".join(miss_team))
                        errors.append(f"{season}: validation failed -> " + " | ".join(msg))
                        continue

                    dst = dest_root / season
                    dst.mkdir(parents=True, exist_ok=True)
                    (dst / "individual_stats_overall.csv").write_bytes(ind_p.read_bytes())
                    (dst / "team_stats.csv").write_bytes(team_p.read_bytes())
                    saved_rows.append((COLLEGE_MAP.get(team_key, team_display_in), season))
                except Exception as e:
                    errors.append(f"{season}: {e}")

            if errors:
                st.warning("Some seasons could not be saved:")
                st.code("\n".join(errors))

            summarise_saved(saved_rows)

    # MODE 3
    else:
        st.caption("Upload a ZIP with structure: `college/season/{individual_stats_overall.csv, team_stats.csv}` (strict).")
        up_zip = st.file_uploader("Upload ZIP", type=["zip"], key="up_zip_multi")

        if st.button("Validate & Save"):
            if not up_zip:
                st.error("Please upload a ZIP file.")
                return

            try:
                zf = zipfile.ZipFile(io.BytesIO(up_zip.getvalue()))
            except Exception as e:
                st.error(f"Could not open ZIP: {e}")
                return

            temp_extract = ART_DIR / "_tmp_extract_multi"
            if temp_extract.exists():
                shutil.rmtree(temp_extract)
            temp_extract.mkdir(parents=True, exist_ok=True)

            extracted = extract_zip_file(zf, temp_extract)

            pairs = {}          # (college_key, season) -> {ind, team}
            ignored_paths = []

            for p in extracted:
                rel = p.relative_to(temp_extract)
                parts = rel.parts
                if len(parts) < 3:
                    ignored_paths.append(str(rel))
                    continue
                college_raw, season, fname = parts[-3], parts[-2], parts[-1]
                fname_l = fname.lower()
                if fname_l not in ("individual_stats_overall.csv", "team_stats.csv"):
                    ignored_paths.append(str(rel))
                    continue
                college_key = norm_key(college_raw)
                entry = pairs.setdefault((college_key, season), {"ind": None, "team": None})
                if fname_l == "individual_stats_overall.csv":
                    entry["ind"] = p
                else:
                    entry["team"] = p

            saved_rows, errors = [], []
            for (college_key, season), files in sorted(pairs.items()):
                ind_p, team_p = files["ind"], files["team"]
                display_college = COLLEGE_MAP.get(college_key, college_key)

                if not ind_p or not team_p:
                    missing = []
                    if not ind_p:  missing.append("individual_stats_overall.csv")
                    if not team_p: missing.append("team_stats.csv")
                    errors.append(f"{display_college} / {season}: missing {', '.join(missing)}.")
                    continue

                try:
                    df_ind = read_slug_csv(ind_p)
                    df_team = read_slug_csv(team_p)
                except Exception as e:
                    errors.append(f"{display_college} / {season}: failed to read CSVs -> {e}")
                    continue

                miss_ind, miss_team = validate_pair(df_ind, df_team)
                if miss_ind or miss_team:
                    msg = []
                    if miss_ind:  msg.append("individual_stats_overall.csv: " + "; ".join(miss_ind))
                    if miss_team: msg.append("team_stats.csv: " + "; ".join(miss_team))
                    errors.append(f"{display_college} / {season}: validation failed -> " + " | ".join(msg))
                    continue

                try:
                    dst = RAW_BASE / college_key / season
                    dst.mkdir(parents=True, exist_ok=True)
                    (dst / "individual_stats_overall.csv").write_bytes(ind_p.read_bytes())
                    (dst / "team_stats.csv").write_bytes(team_p.read_bytes())
                    saved_rows.append((display_college, season))
                except Exception as e:
                    errors.append(f"{display_college} / {season}: failed to save -> {e}")

            if ignored_paths:
                st.info("Ignored files/folders not matching `college/season/file.csv` (extra root folders are fine):")
                st.code("\n".join(sorted(ignored_paths)))

            if errors:
                st.warning("Some (college, season) pairs could not be saved:")
                st.code("\n".join(errors))

            summarise_saved(saved_rows)

    st.markdown("---")
    from run_pipeline import run_pipeline
    if st.button("Run pipeline now on all data"):
        rid = run_pipeline()
        if rid:
            st.success(f"Pipeline completed: {rid}")
            st.info("Click Finish button to update results.")

    if st.button("Finish"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Refreshed. Go to the 'Roster' tab to view new data.")
