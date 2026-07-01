from typing import Literal, assert_never

import streamlit as st
from passlib.hash import pbkdf2_sha256
from streamlit.navigation.page import StreamlitPage

from .._state import state
from ..models import (
    ActiveTask,
    ExplainChallenge,
    RateChallenge,
    User,
    UserKind,
    UserRole,
)
from ..storage_sqlite import SqliteStorage

PageKey = Literal["login", "challenge", "task", "thanks"]
CHALLENGE_NAMES = {
    "explain_dummy": "Dummy",
    "explain_butterfly_easy": "Butterfly Wings (Easy)",
    "explain_butterfly_difficult": "Butterfly Wings (Difficult)",
    "rate_dummy": "Dummy",
    "rate_butterfly_easy": "Butterfly (Easy)",
    "rate_butterfly_difficult": "Butterfly (Difficult)",
}


class PageBuilder:
    def __init__(self) -> None:

        self.storage = SqliteStorage()
        if not state.user:
            self.pages: dict[PageKey, StreamlitPage] = {
                "login": st.Page(
                    self.render_login_page,
                    title="Login",
                    icon=":material/login:",
                    url_path="login",
                    default=True,
                )
            }
        else:
            self.pages = {
                "challenge": st.Page(
                    self.render_challenge_selection_page,
                    title="Select challenge",
                    icon=":material/list_alt:",
                    url_path="challenge",
                    default=True,
                ),
                "task": st.Page(
                    self.render_task_page,
                    title="Current task",
                    icon=":material/psychology_alt:",
                    url_path="task",
                ),
            }

    def render_login_page(self) -> None:
        st.title(":butterfly: Welcome to SpeciFly!")
        st.subheader("Can you tell butterfly species apart?")
        if state.user:
            st.info(f"Logged in as {state.user}.")
            if st.button("Logout"):
                state.reset()
                st.rerun()

            if st.button("Select challenge"):
                self.switch_to("challenge")

            return

        st.info("Please create an account or login.")
        login_tab, create_tab = st.tabs(
            ["Login", "Create account"], key="login_create_tabs", on_change="rerun"
        )
        with login_tab, st.form("login_form"):
            typed_user_id = st.text_input(
                "Username",
                key="login_username",
                max_chars=32,
                icon=":material/person:",
            )
            typed_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                max_chars=100,
                icon=":material/lock:",
            )

            if st.form_submit_button("Login"):
                user = self.storage.fetch_user(typed_user_id)
                if user is None:
                    st.error(
                        f"User '{typed_user_id}' not found. Please create an account."
                    )
                elif pbkdf2_sha256.verify(typed_password, user.hashed_password):
                    st.toast(f"Welcome back, {user.id}!")
                    state.user = user
                    self.switch_to("challenge")
                else:
                    st.error("Incorrect password.")

        st.write(f"typed username: {typed_user_id}")
        with create_tab, st.form("create_form"):
            if create_tab.open:
                typed_new_user_id = st.text_input(
                    "Username",
                    value=typed_user_id,
                    key="create_username",
                    max_chars=32,
                    icon=":material/person:",
                )

                typed_new_password = st.text_input(
                    "Password",
                    value=typed_password,
                    type="password",
                    key="create_password",
                    max_chars=100,
                    icon=":material/lock:",
                )

                retyped_new_password = st.text_input(
                    "Retype Password",
                    type="password",
                    key="retyped_password",
                    max_chars=100,
                    icon=":material/lock:",
                )
            else:
                typed_new_user_id = None
                typed_new_password = None
                retyped_new_password = None

            if (
                st.form_submit_button("Create account", disabled=not typed_new_user_id)
                and typed_new_user_id
            ):
                if not typed_new_password:
                    st.error("Please enter a password.")
                    return

                if typed_new_password != retyped_new_password:
                    st.error("Passwords do not match.")
                    return

                user = User(
                    id=typed_new_user_id,
                    kind=UserKind.HUMAN,
                    role=UserRole.PARTICIPANT,
                    hashed_password=pbkdf2_sha256.hash(typed_new_password),
                )
                self.storage.add_user(user)
                st.success(f"Account created for {user.id}!")
                state.user = user
                return

    def render_challenge_selection_page(self) -> None:
        user = self.get_user()
        st.header("Choose a challenge")
        challenge_data = self.storage.fetch_challenges(user)
        if not challenge_data.explain_challenges:
            st.error("No challenges found.")
            st.stop()

        explain_col, rate_col = st.columns(2)
        with explain_col:
            st.subheader("Detect differences")
        with rate_col:
            st.subheader("Rate differences")

        for (
            explain_challenge_id,
            explain_challenge,
        ) in challenge_data.explain_challenges.items():
            explain_col, rate_col = st.columns(2)
            with explain_col:
                if st.button(
                    CHALLENGE_NAMES[explain_challenge_id],
                    key=explain_challenge_id,
                    disabled=explain_challenge.finished,
                ):
                    state.task = ActiveTask(
                        challenge_data=challenge_data,
                        challenge_id=explain_challenge_id,
                        task_idx=challenge_data.explain_challenges[
                            explain_challenge_id
                        ].first_undone
                        or 0,
                    )
                    self.switch_to("task")
                st.progress(
                    explain_challenge.progress,
                    text=f"{explain_challenge.done_count}/{explain_challenge.task_count}",
                )

            if explain_challenge_id == "explain_dummy":
                rate_challenge_id = "rate_dummy"
            elif explain_challenge_id == "explain_butterfly_easy":
                rate_challenge_id = "rate_butterfly_easy"
            elif explain_challenge_id == "explain_butterfly_difficult":
                rate_challenge_id = "rate_butterfly_difficult"
            else:
                assert_never(explain_challenge_id)

            rate_challenge = challenge_data.rate_challenges.get(rate_challenge_id)
            with rate_col:
                if (
                    st.button(
                        CHALLENGE_NAMES[rate_challenge_id],
                        key=rate_challenge_id,
                        disabled=rate_challenge is None or rate_challenge.finished,
                    )
                    and rate_challenge is not None
                ):
                    state.task = ActiveTask(
                        challenge_data=challenge_data,
                        challenge_id=rate_challenge_id,
                        task_idx=rate_challenge.first_undone or 0,
                    )
                    self.switch_to("task")
                st.progress(
                    0 if rate_challenge is None else rate_challenge.progress,
                    text=f"{0 if rate_challenge is None else rate_challenge.done_count}/{explain_challenge.task_count if rate_challenge is None else rate_challenge.task_count}",
                )

    def render_task_page(self) -> None:
        task = state.task
        if not task:
            st.toast("Please select a challenge first.")
            self.switch_to("challenge")

        if task.challenge_id in task.challenge_data.explain_challenges:
            challenge = task.challenge_data.explain_challenges[task.challenge_id]
            self._render_explain_task(task, challenge)
        elif task.challenge_id in task.challenge_data.rate_challenges:
            challenge = task.challenge_data.rate_challenges[task.challenge_id]
            self._render_rate_task(task, challenge)
        else:
            st.error("Challenge state not found.")
            self.switch_to("challenge")

        bottom_left, bottom_center, bottom_right = st.columns([0.1, 0.8, 0.1])
        with bottom_left:
            if task.task_idx > 0:
                if st.button("Previous task"):
                    task.task_idx -= 1
                    st.rerun()

        with bottom_right:
            if task.task_idx < challenge.task_count - 1:
                if st.button("Next task"):
                    task.task_idx += 1
                    st.rerun()
            else:
                if st.button("Finish challenge"):
                    st.toast(
                        f"Thank you for finishing {CHALLENGE_NAMES[challenge.id]}!"
                    )
                    st.balloons()
                    self.switch_to("challenge")

    def _render_explain_task(
        self, task: ActiveTask, challenge: ExplainChallenge
    ) -> None:
        st.error("Explain task rendering not implemented yet.")

    def _render_rate_task(self, task: ActiveTask, challenge: RateChallenge) -> None:
        st.error("Rate task rendering not implemented yet.")

    def get_user(self) -> User:
        if not state.user:
            self.switch_to("login")

        user = state.user
        if not user:
            st.error("A named account is required for this study.")
            st.stop()

        return user

    def switch_to(self, page_key: PageKey):
        page = self.pages.get(page_key)
        if page is None:
            st.rerun()
        st.switch_page(page)
