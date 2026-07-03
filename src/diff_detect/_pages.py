from typing import Callable

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from ._challenge_selection_page import render_challenge_selection_page
from ._login_page import auto_login as render_login_page
from ._page_utils import PageKey
from ._state import state
from ._task_page import render_task_page

st.set_page_config(page_title="SpeciFly", page_icon=":butterfly:")


def wrap_page_rendering(page: Callable[[], PageKey | None]) -> Callable[[], None]:
    """Wraps the page rendering function to handle page switching."""

    def wrapper() -> None:
        page_key = page()
        if page_key is not None:
            next_page = PAGES.get(page_key)
            if next_page is None:
                st.rerun()

            st.switch_page(next_page)

    return wrapper


PAGES: dict[PageKey, StreamlitPage] = {
    "login": st.Page(
        wrap_page_rendering(render_login_page),
        title="Login",
        icon=":material/login:",
        url_path="login",
        default=state.user is None,
    ),
    "challenge": st.Page(
        wrap_page_rendering(render_challenge_selection_page),
        title="Select challenge",
        icon=":material/list_alt:",
        url_path="challenge",
        default=state.user is not None,
    ),
    "task": st.Page(
        wrap_page_rendering(render_task_page),
        title="Current task",
        icon=":material/psychology_alt:",
        url_path="task",
    ),
}
