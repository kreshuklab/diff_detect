import streamlit as st

from ._pages import PAGES
from ._state import state


def main() -> None:

    if state.user is not None:
        with st.sidebar:
            st.info(f"Logged in as {state.user.id}.")
            st.button(
                "Logout",
                on_click=state.reset,
                type="secondary",
                width="stretch",
                icon=":material/logout:",
            )

    if state.toaster:
        # when streamlit reruns toast messages disappear immediately,
        # so we put them in the toaster instead if we intend to rerun and render them here instead
        st.toast(state.toaster)
        del state.toaster

    page = st.navigation(list(PAGES.values()), position="sidebar", expanded=True)
    page.run()
