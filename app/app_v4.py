import streamlit as st
from tabs import train_explore, historical_data, matchups, uploads

st.set_page_config(page_title="Coach Scouting Dashboard", layout="wide")

tab_train, tab_hist, tab_matchups, tab_upload = st.tabs(
    ["Train & Explore", "Team & Player Data", "Match-ups", "Upload & Classify"]
)

with tab_train:
    train_explore.render()

with tab_hist:
    historical_data.render()

with tab_matchups:
    matchups.render()

with tab_upload:
    uploads.render()
