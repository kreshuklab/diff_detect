import streamlit as st
from st_login_form import login_form
from st_supabase_connection import SupabaseConnection


def configured_login_form(supabase: SupabaseConnection):
    try:
        _ = login_form(
            title="Account",
            icon=":material/lock:",
            allow_guest=False,
            create_title="Create an account",
            login_title="Login",
            create_submit_label="Create account",
            login_submit_label="Login",
            constrain_password=False,
            supabase_connection=supabase,
        )
    except Exception as exc:
        message = str(exc)
        if "public.users" in message or "PGRST205" in message:
            st.error(
                "Supabase is connected, but the `public.users` table is missing. "
                "Run `schema/supabase.sql` in the Supabase SQL editor."
            )
        else:
            st.error(
                "Login is not configured yet. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in Supabase credentials."
            )
        st.exception(exc)
        return
