import streamlit as st

from .pages import build_pages


def main() -> None:
    page = st.navigation(build_pages(), position="sidebar", expanded=True)
    page.run()
