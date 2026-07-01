import streamlit as st
from streamlit.navigation.page import StreamlitPage


def switch_to(pages: dict[str, StreamlitPage], page_key: str) -> None:
    page = pages.get(page_key)
    if page is None:
        st.rerun()
    st.switch_page(page)
