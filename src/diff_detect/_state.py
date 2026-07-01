import streamlit as st

from .models import (
    ActiveTask,
    User,
)


class SessionState:
    @property
    def task(self) -> ActiveTask | None:
        return st.session_state.get("task")

    @task.setter
    def task(self, value: ActiveTask | None) -> None:
        st.session_state["task"] = value

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


state = SessionState()
