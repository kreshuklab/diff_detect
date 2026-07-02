import streamlit as st

from .models import ActiveExplainChallenge, ActiveRateChallenge, User


class SessionState:
    @property
    def active_challenge(self) -> ActiveExplainChallenge | ActiveRateChallenge | None:
        return st.session_state.get("active_challenge")

    @active_challenge.setter
    def active_challenge(
        self, value: ActiveExplainChallenge | ActiveRateChallenge | None
    ) -> None:
        st.session_state["active_challenge"] = value

    @property
    def task_idx(self) -> int:
        return st.session_state.get("task_idx") or 0

    @task_idx.setter
    def task_idx(self, value: int) -> None:
        st.session_state["task_idx"] = value

    @task_idx.deleter
    def task_idx(self) -> None:
        del st.session_state["task_idx"]

    @property
    def toaster(self) -> str | None:
        return st.session_state.get("toaster")

    @toaster.setter
    def toaster(self, value: str | None) -> None:
        st.session_state["toaster"] = value

    @toaster.deleter
    def toaster(self) -> None:
        del st.session_state["toaster"]

    @property
    def user(self) -> User | None:
        return st.session_state.get("user")

    @user.setter
    def user(self, value: User) -> None:
        st.session_state["user"] = value

    @user.deleter
    def user(self) -> None:
        del st.session_state["user"]

    def reset(self):
        if "user" in st.session_state:
            del st.session_state["user"]
        if "task" in st.session_state:
            del st.session_state["task"]
        if "task_idx" in st.session_state:
            del st.session_state["task_idx"]


state = SessionState()
