import streamlit as st
from passlib.hash import pbkdf2_sha256

from diff_detect._page_utils import PageKey

from ._state import state
from ._storage import storage
from .models import User, UserKind, UserRole


def auto_login() -> PageKey | None:
    username = "dev"
    user = storage.fetch_user(username)
    if user is None:
        user = User(
            name=username,
            id=username,
            lab=None,
            kind=UserKind.HUMAN,
            role=UserRole.MAINTAINER,
            hashed_password=None,
        )
        storage.add_user(user)

    state.toaster = f"Welcome back, {user.name}!"
    state.user = user
    return "challenge"


def render_login_page() -> PageKey | None:
    st.set_page_config(layout="centered")
    st.title(":butterfly: Welcome to Butterfly Detective!")
    st.subheader("Can you tell butterfly species apart?")
    if state.user:
        st.info(f"Logged in as {state.user.id}.")
        left, right = st.columns(2)
        with left:
            if st.button("Logout", width="stretch", type="secondary"):
                state.reset()
                st.rerun()
        with right:
            if st.button("Select challenge", width="stretch"):
                return "challenge"

        return

    st.info("Please create an account or login.")
    lab_options = storage.fetch_lab_options()
    create_tab, login_tab = st.tabs(
        ["Create account", "Login"], key="login_create_tabs", on_change="rerun"
    )
    with create_tab, st.form("create_form", enter_to_submit=False):
        typed_user_name = st.text_input(
            "Username",
            key="create_username",
            max_chars=32,
            icon=":material/person:",
        )
        typed_lab = st.selectbox(
            "Lab",
            options=lab_options,
            index=None,
            key="create_lab",
            accept_new_options=True,
        )
        if st.form_submit_button("Create account"):
            if not typed_user_name:
                st.error("Please enter a username.")
                return
            if typed_user_name.lower() == "ai" or typed_user_name.lower().startswith(
                "ai_"
            ):
                st.error("Username cannot be 'ai' or start with 'ai_'.")
                return
            if typed_user_name.lower().startswith("dummyuser"):
                st.error("Username cannot start with 'dummyuser'.")
                return
            if "/" in typed_user_name:
                st.error("Username cannot contain slashes.")
                return

            user_id = _get_user_id(typed_user_name, typed_lab)

            user = storage.fetch_user(user_id)
            if user is not None:
                st.error(
                    f"User ID '{user_id}' already exists.\nPlease choose a different username or lab or use the login tab if you already have an account."
                )
                return

            user = User(
                id=user_id,
                name=typed_user_name,
                lab=typed_lab,
                kind=UserKind.HUMAN,
                role=UserRole.PARTICIPANT,
                hashed_password=None,
            )
            state.toaster = f"Welcome {user.name}!"
            storage.add_user(user)
            from_lab = "" if user.lab is None else f" from {user.lab}"
            st.success(f"Account created for {user.name}{from_lab}!")
            state.user = user
            return "challenge"

    with login_tab, st.form("login_form", enter_to_submit=False):
        if login_tab.open:
            typed_user_name = st.text_input(
                "Username",
                value=typed_user_name,
                key="login_username",
                max_chars=32,
                icon=":material/person:",
            )
            typed_lab = st.selectbox(
                "Lab",
                options=lab_options,
                index=None,
                key="login_lab",
                # max_chars=32,
                # icon=":material/science:",
                accept_new_options=True,
            )

            user_id = _get_user_id(typed_user_name, typed_lab)
            if st.form_submit_button("Login"):
                user = storage.fetch_user(user_id)
                if user is None:
                    from_lab = f" from '{typed_lab}'" if typed_lab else ""
                    st.error(
                        f"User '{typed_user_name}'{from_lab} not found. Please create an account."
                    )
                else:
                    state.toaster = f"Welcome back, {user.name}!"
                    state.user = user
                    return "challenge"


def render_login_page_with_password() -> PageKey | None:
    st.set_page_config(layout="centered")
    st.title(":butterfly: Welcome to Butterfly Detective!")
    st.subheader("Can you tell apart butterfly species by their wings?")
    if state.user:
        st.info(f"Logged in as {state.user.id}.")
        left, right = st.columns(2)
        with left:
            if st.button("Logout", width="stretch", type="secondary"):
                state.reset()
                st.rerun()
        with right:
            if st.button("Select challenge", width="stretch"):
                return "challenge"

        return

    st.info("Please create an account or login.")
    login_tab, create_tab = st.tabs(
        ["Login", "Create account"], key="login_create_tabs", on_change="rerun"
    )
    with login_tab, st.form("login_form", enter_to_submit=False):
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
            user = storage.fetch_user(typed_user_id)
            if user is None:
                st.error(f"User '{typed_user_id}' not found. Please create an account.")
            elif user.hashed_password is None or pbkdf2_sha256.verify(
                typed_password, user.hashed_password
            ):
                state.toaster = f"Welcome back, {user.name}!"

                state.user = user
                return "challenge"
            else:
                st.error("Incorrect password.")

    with create_tab, st.form("create_form", enter_to_submit=False):
        if create_tab.open:
            typed_new_user_id = st.text_input(
                "Username",
                value=typed_user_id,
                key="create_username",
                max_chars=32,
                icon=":material/person:",
            )

            lab = st.selectbox(
                "Lab",
                options=["Kreshuklab"],
                index=None,
                key="create_lab",
                # max_chars=32,
                # icon=":material/science:",
                accept_new_options=True,
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
                icon=":material/enhanced_encryption:",
            )
        else:
            typed_new_user_id = None
            lab = None
            typed_new_password = None
            retyped_new_password = None

        if st.form_submit_button("Create account"):
            if not typed_new_user_id:
                st.error("Please enter a username.")
                return

            user = storage.fetch_user(typed_new_user_id)
            if user is not None:
                st.error("Username already exists.")
                return

            if not typed_new_password:
                st.error("Please enter a password.")
                return

            if typed_new_password != retyped_new_password:
                st.error("Passwords do not match.")
                return

            user = User(
                id=typed_new_user_id,
                name=typed_new_user_id,
                lab=lab or "Guest",
                kind=UserKind.HUMAN,
                role=UserRole.PARTICIPANT,
                hashed_password=pbkdf2_sha256.hash(typed_new_password),
            )
            state.toaster = f"Welcome {user.id}!"
            storage.add_user(user)
            st.success(f"Account created for {typed_new_user_id}!")
            state.user = user
            return "challenge"


def _get_user_id(username: str, lab: str | None) -> str:
    return f"{lab}/{username}" if lab else username


# def configured_login_form(supabase: SupabaseConnection):
#     try:
#         _ = login_form(
#             title="Account",
#             icon=":material/lock:",
#             allow_guest=False,
#             create_title="Create an account",
#             login_title="Login",
#             create_submit_label="Create account",
#             login_submit_label="Login",
#             constrain_password=False,
#             supabase_connection=supabase,
#         )
#     except Exception as exc:
#         message = str(exc)
#         if "public.users" in message or "PGRST205" in message:
#             st.error(
#                 "Supabase is connected, but the `public.users` table is missing. "
#                 "Run `schema/supabase.sql` in the Supabase SQL editor."
#             )
#         else:
#             st.error(
#                 "Login is not configured yet. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in Supabase credentials."
#             )
#         st.exception(exc)
#         return
