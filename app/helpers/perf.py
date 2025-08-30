# helpers/perf.py
import time
import contextlib
import streamlit as st

@contextlib.contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        st.caption(f"⏱ {label}: {dt:.1f} ms")
