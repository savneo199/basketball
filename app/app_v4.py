import streamlit as st
from tabs import train_explore, historical_data, uploads, player_finder, compare_player

st.set_page_config(page_title="Coach Scouting Dashboard", layout="wide")

tab_train, tab_hist, tab_upload = st.tabs(
    ["Train & Explore", "Team & Player Data", "Upload & Classify"]
)

with tab_train:
    train_explore.render()

with tab_hist:
    historical_data.render()

# with tab_finder:
#     player_finder.render()

# with tab_compare:
#     compare_player.render()

with tab_upload:
    uploads.render()





