# Data-Driven Framework for Athlete Profiling

This repository contains the implementation of the MSc Artificial Intelligence final project at Imperial College London: **Data-Driven Framework for Athlete Profiling**, by Savvas Neofytou (2025).  

The project develops an end-to-end, machine learning pipeline for **role-based profiling of NCAA women’s basketball players**, offering an interpretable, reproducible, and deployable tool for coaches and analysts.

## Project Overview

- **Dataset**: 2,065 player-season observations from NCAA Women’s Basketball (2002–2024).  
- **Features**: 81 pace-adjusted and per-40 metrics engineered from box-score statistics.  
- **Clustering**: Multiple algorithms evaluated (K-Means, GMM, Agglomerative, HDBSCAN). Final model: **K-Means with 6 clusters**, identifying nuanced archetypes such as:  
  - Floor General  
  - Two-Way Guard  
  - Deep Range Specialist  
  - Rim Protector / Rebounding Big  
  - Non-Impact Role Player  
- **Explainability**: Surrogate Random Forest with SHAP attributions highlights key discriminators (USG%, AST%, TOV%, eFG%, BLK%).  
- **Deployment**: A **Streamlit dashboard** supports roster scanning, player comparison, lineup previews, and transparency into statistical drivers.  

This framework bridges data science and practical coaching needs by providing **interpretable and actionable insights** for roster building, opponent scouting, and lineup optimisation.

## Repository Structure (only main files shown)
```
├── app/
│    ├── helpers
│    │   ├── archetype_positions.py
│    │   ├── court_builder.py
│    │   └── helpers.py
│    ├── tabs
│    │   ├── __pycache__/
│    │   ├── __init__.py
│    │   ├── archetypes.py
│    │   ├── explain.py
│    │   ├── historical_data.py
│    │   ├── home.py
│    │   ├── train_explore.py
│    │   └── uploads.py
│    ├── app.py
│    ├── get_paths.py
│    └── run_pipeline.py
├── artifacts/
├── college_basketball_dataset/
├── data/
├── notebooks/
│    ├── scraper.ipynb
│    ├── preprocess.ipynb
│    ├── explore.ipynb
│    ├── clustering_experimentation.ipynb
│    └── k_means_final.ipynb
├── pipeline/
│    ├── orchestrate.py
│    ├── notebook_exec.py
│    └── config.yaml
├── .DS_Store
├── .dockerignore
├── .gitignore
├── Dockerfile
└── requirements.txt
```
**Map**
- `scraper.ipynb` → Web scraping pipeline for NCAA player statistics.  
- `preprocess.ipynb` → Data cleaning, standardisation, and feature engineering.  
- `explore.ipynb` → Exploratory data analysis and PCA-based dimensionality reduction.  
- `clustering_experimentation.ipynb` → Comparison of clustering algorithms.  
- `k_means_final.ipynb` → Final K-Means implementation and archetype assignment.  
- `dashboard/` → Streamlit app prototype for interactive use.  
- `data/` → Processed datasets (per-season and aggregated).  

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://gitlab.doc.ic.ac.uk/sn1624/data-driven-framework-for-athlete-profiling.git
   cd basketball

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Run the Streamlit dashboard to visualise results:
  ```bash
  streamlit run dashboard/app.py
  ```

4. Explore notebooks (.ipynb) for end-to-end data pipeline and clustering experiments.

## Citation

If you use this work, please cite:
Savvas Neofytou. Data-Driven Framework for Athlete Profiling. MSc Artificial Intelligence Thesis, Imperial College London, 2025.
