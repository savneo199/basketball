
import streamlit as st

def render():
    st.header("AI-driven player archetyping tool for NCAA Women’s Basketball")

    st.markdown(
        """
    **WCBB Coach** helps you quickly understand **player roles and styles** across NCAA women’s basketball.  
    We use game data to group players into **archetypes** (e.g., rim protectors, spot-up wings, 
    on-ball creators), so you can scan a roster, find comparable players, and spot lineup fits faster.
    """
    )

    st.subheader("What you'll find in the tabs")
    st.markdown(
            """
    - **Train & Explore:** Archetype Model Breakdown and the ability to train it on new data and refresh the database.
    - **Team & Player Finder:** See team and player stats and archetypes season by season, compare player's performance and statistics,
    and preview predicted lineups.
    - **Archetypes:** Read short definitions, key stats, and typical strengths/limitations for each archetype.  
    - **Upload & Classify:** Upload and classify new player data to improve archetype model accuracy, and preview the 
    archetypes of uploaded players.
    """
        )