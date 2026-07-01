import streamlit as st
from st_supabase_connection import SupabaseConnection

from diff_detect.app_old import load_challenge_progress, progress_label
from diff_detect.storage import SupabaseStorage

from ._login import configured_login_form


class PageBuilder:
    def __init__(self) -> None:
        self.supabase = st.connection(name="supabase", type=SupabaseConnection)
        self.storage = SupabaseStorage(self.supabase)
        if not st.session_state.get("authenticated", False):
            self.pages = {
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
                    self.render_challenge_page,
                    title="Select challenge",
                    icon=":material/list_alt:",
                    url_path="challenge",
                    default=True,
                ),
                "selection": st.Page(
                    self.render_selection_tasks_page,
                    title="Selection tasks",
                    icon=":material/rule:",
                    url_path="selection",
                ),
                "rating": st.Page(
                    self.render_rating_tasks_page,
                    title="Rating tasks",
                    icon=":material/rate_review:",
                    url_path="rating",
                ),
                "thanks": st.Page(
                    self.render_thank_you_page,
                    title="Thank you",
                    icon=":material/check_circle:",
                    url_path="thanks",
                ),
            }

    @property
    def authenticated(self) -> bool:
        return st.session_state.get("authenticated", False)

    @property
    def username(self) -> str:
        if not self.authenticated:
            self.switch_to("login")
        username = st.session_state.get("username")
        if not isinstance(username, str) or not username:
            st.error("A named account is required.")
            st.stop()
        return username

    def render_login_page(self) -> None:
        st.title(":butterfly: Welcome to SpeciFly!")
        st.subheader("Can you tell butterfly species apart?")
        st.write("Please create an account or login.")
        configured_login_form(self.supabase)

    def render_challenge_page(self) -> None:
        st.header("Choose a challenge")
        challenges = self.storage.fetch_challenge_progress
        if not challenges:
            st.error("No challenges found.")
            st.stop()

        try:
            progress = load_challenge_progress(self.supabase, self.username, challenges)
        except Exception as exc:
            st.error("Failed to load challenge progress.")
            st.exception(exc)
            return

        progress_by_key = {item.key: item for item in progress}
        st.dataframe(
            [
                {
                    "Dataset": item.dataset_id,
                    "Challenge": item.challenge_id,
                    "Tasks": item.task_count,
                    "Selections": progress_label(item.submitted_count, item.task_count),
                    "Ratings": progress_label(item.rated_count, item.task_count),
                    "Status": challenge_status(item),
                }
                for item in progress
            ],
            hide_index=True,
            use_container_width=True,
        )

        selectable_keys = [item.key for item in progress if item.is_selectable]
        if not selectable_keys:
            st.error("No playable challenges are available.")
            return

        preferred_key = preferred_selectable_challenge_key(selectable_keys)
        selected_key = st.radio(
            "Challenge",
            selectable_keys,
            index=selectable_keys.index(preferred_key),
            format_func=lambda key: challenge_option_label(progress_by_key[key]),
        )
        if st.button("Continue", type="primary"):
            selected_progress = progress_by_key[selected_key]
            select_challenge(
                selected_progress.dataset_id, selected_progress.challenge_id
            )
            switch_to(next_page_for_progress(selected_progress))

    def switch_to(self, page_key: str) -> None:
        page = self.pages.get(page_key)
        if page is None:
            st.rerun()
        st.switch_page(page)
