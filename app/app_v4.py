import streamlit as st
from tabs import train_explore, historical_data, uploads, home

st.set_page_config(page_title="Coach Scouting Dashboard", layout="wide")

tab_home, tab_train, tab_hist, tab_upload = st.tabs(
    ["Home", "Train & Explore", "Team & Player Data", "Upload & Classify"]
)

with tab_home:
    home.render()

with tab_train:
    train_explore.render()

with tab_hist:
    historical_data.render()

with tab_upload:
    uploads.render()





