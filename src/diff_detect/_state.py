import streamlit as st

from .models import (
    ActiveTask,
    User,
)


class SessionState:
    # @property
    # def typed_user_id(self) -> str | None:
    #     return st.session_state.get("typed_user_id")

    # @typed_user_id.setter
    # def typed_user_id(self, value: str) -> None:
    #     st.session_state["typed_user_id"] = value

    # @typed_user_id.deleter
    # def typed_user_id(self) -> None:
    #     del st.session_state["typed_user_id"]

    @property
    def user(self) -> User | None:
        return st.session_state.get("user")

    @user.setter
    def user(self, value: User) -> None:
        st.session_state["user"] = value

    @user.deleter
    def user(self) -> None:
        del st.session_state["user"]

    @property
    def task(self) -> ActiveTask | None:
        return st.session_state.get("task")

    @task.setter
    def task(self, value: ActiveTask | None) -> None:
        st.session_state["task"] = value

    def reset(self):
        if "user" in st.session_state:
            del st.session_state["user"]
        if "task" in st.session_state:
            del st.session_state["task"]


state = SessionState()
