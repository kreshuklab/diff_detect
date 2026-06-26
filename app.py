from __future__ import annotations

import json
from typing import Any, cast

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from study_storage import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    LABEL_STYLES,
    MIN_ROUNDS,
    build_rating_options,
    canvas_has_objects,
    choose_rounds,
    composite_annotation,
    decode_png,
    fetch_peer_submission,
    fetch_user_ratings,
    fetch_user_submissions,
    image_for_id,
    load_rounds,
    load_seeded_annotations,
    load_wing_image,
    reference_images,
    upsert_rating,
    upsert_submission,
)

try:
    from st_login_form import login_form
except ModuleNotFoundError:
    login_form = None


st.set_page_config(page_title="Butterfly Wing Study", page_icon=":butterfly:", layout="wide")


def init_state() -> None:
    defaults: dict[str, Any] = {
        "phase": "intro",
        "round_index": 0,
        "selected_image_id": None,
        "completed_intro": False,
        "rating_intro_seen": False,
        "storage_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    st.title("Butterfly Wing Study")
    st.write("Hello butterfly-wing-expert-to-be, please create an account or login!")

    if login_form is None:
        st.error("Missing dependency: `st-login-form`. Install dependencies with `pip install -r requirements.txt`.")
        return

    try:
        supabase = login_form(
            title="Account",
            icon=":material/lock:",
            allow_guest=False,
            create_title="Create an account",
            login_title="Login",
            create_submit_label="Create account",
            login_submit_label="Login",
        )
    except Exception as exc:
        st.error("Login is not configured yet. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in Supabase credentials.")
        st.exception(exc)
        return

    if not st.session_state.get("authenticated"):
        return

    username = st.session_state.get("username")
    if not username:
        st.error("A named account is required for this study.")
        return

    init_state()
    rounds = choose_rounds(load_rounds(), username, MIN_ROUNDS)
    seeded_annotations = load_seeded_annotations()

    try:
        submissions = fetch_user_submissions(supabase, username)
        ratings = fetch_user_ratings(supabase, username)
    except Exception as exc:
        st.error("Could not load study progress from Supabase. Check that `schema/supabase.sql` has been applied.")
        st.exception(exc)
        return

    submitted_task_ids = {submission["task_id"] for submission in submissions}
    rated_task_ids = {rating["task_id"] for rating in ratings}

    if not st.session_state.completed_intro:
        render_intro(username)
    elif len(submitted_task_ids) < len(rounds):
        render_selection_or_annotation(supabase, username, rounds, submissions)
    elif not st.session_state.rating_intro_seen:
        render_rating_intro(len(submissions))
    elif len(rated_task_ids) < len(rounds):
        render_rating(supabase, username, rounds, submissions, rated_task_ids, seeded_annotations)
    else:
        render_done()


def render_intro(username: str) -> None:
    st.header(f"Hello butterfly wing expert {username}!")
    st.write(
        "You will be shown four sets of butterfly wings, three of which are from the same species. "
        "Please select the one from a different species by clicking on it."
    )
    st.write(
        "Do not choose the odd one out by a damaged wing or any other non-biological difference. "
        f"After you choose, label your selection to explain why you think it is from a different species. "
        f"Repeat this for {MIN_ROUNDS} rounds."
    )
    if st.button("OK", type="primary"):
        st.session_state.completed_intro = True
        st.rerun()


def render_selection_or_annotation(
    supabase: Any,
    username: str,
    rounds: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
) -> None:
    submitted = {submission["task_id"] for submission in submissions}
    next_round = next(task for task in rounds if task["task_id"] not in submitted)
    progress_index = len(submitted) + 1

    st.caption(f"Round {progress_index} of {len(rounds)}")
    if not st.session_state.selected_image_id:
        render_selection(next_round)
    else:
        render_annotation(supabase, username, next_round)


def render_selection(task: dict[str, Any]) -> None:
    st.header("Please select the one of a different species than the other three.")
    shuffled_images = list(task["images"])
    columns = st.columns(4)
    for index, image_spec in enumerate(shuffled_images):
        with columns[index]:
            image = load_wing_image(image_spec, size=(360, 250))
            st.image(image, use_container_width=True)
            if st.button("Select", key=f"select_{task['task_id']}_{image_spec['image_id']}", use_container_width=True):
                st.session_state.selected_image_id = image_spec["image_id"]
                st.rerun()


def render_annotation(supabase: Any, username: str, task: dict[str, Any]) -> None:
    selected_id = st.session_state.selected_image_id
    selected_spec = image_for_id(task, selected_id)
    selected_image = load_wing_image(selected_spec)
    references = reference_images(task, selected_id)

    st.header("Annotate the differences in the selected one.")
    left, right = st.columns([3, 1])

    with right:
        st.subheader("References")
        for reference in references:
            st.image(load_wing_image(reference, size=(220, 140)), use_container_width=True)

    with left:
        label = st.radio("Difference label", ["shape", "color", "texture"], horizontal=True)
        style = LABEL_STYLES[label]
        canvas_result = st_canvas(
            fill_color=style["fill"],
            stroke_width=8,
            stroke_color=style["color"],
            background_image=cast(Any, selected_image),
            update_streamlit=True,
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="freedraw",
            display_toolbar=True,
            key=f"canvas_{task['task_id']}_{selected_id}_{label}",
        )
        explanation = st.text_area("Optional explanation", placeholder="Briefly describe the visible biological difference.")

        back_col, save_col = st.columns([1, 2])
        with back_col:
            if st.button("Choose another image"):
                st.session_state.selected_image_id = None
                st.rerun()
        with save_col:
            if st.button("Next", type="primary", use_container_width=True):
                if not canvas_has_objects(canvas_result.json_data):
                    st.warning("Please add at least one annotation stroke before continuing.")
                    return

                composite = composite_annotation(selected_image, canvas_result.image_data)
                payload = {
                    "username": username,
                    "task_id": task["task_id"],
                    "selected_image_id": selected_id,
                    "label": label,
                    "explanation": explanation.strip() or None,
                    "canvas_json": canvas_result.json_data,
                    "composite_png_base64": composite,
                }
                try:
                    upsert_submission(supabase, payload)
                except Exception as exc:
                    st.error("Could not save annotation to Supabase.")
                    st.exception(exc)
                    return
                st.session_state.selected_image_id = None
                st.rerun()


def render_rating_intro(completed_count: int) -> None:
    st.header(f"Thank you for explaining {completed_count} selections!")
    st.write("Now, please take some time to rate others' choices.")
    if st.button("Next", type="primary"):
        st.session_state.rating_intro_seen = True
        st.rerun()


def render_rating(
    supabase: Any,
    username: str,
    rounds: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
    rated_task_ids: set[str],
    seeded_annotations: list[dict[str, Any]],
) -> None:
    submissions_by_task = {submission["task_id"]: submission for submission in submissions}
    task = next(task for task in rounds if task["task_id"] not in rated_task_ids)
    own_submission = submissions_by_task[task["task_id"]]
    peer_submission = fetch_peer_submission(supabase, username, task["task_id"])
    options = build_rating_options(task, own_submission, peer_submission, seeded_annotations, username)

    st.header("For this set of butterfly wings, choose who selected the single species most convincingly.")
    cols = st.columns(4)
    for index, image_spec in enumerate(task["images"]):
        with cols[index]:
            st.image(load_wing_image(image_spec, size=(260, 180)), use_container_width=True)

    st.divider()
    option_cols = st.columns(len(options))
    for index, option in enumerate(options):
        with option_cols[index]:
            st.image(decode_png(option.composite_png_base64), use_container_width=True)
            st.caption(f"Label: {option.label}")
            if option.explanation:
                st.write(option.explanation)
            if st.button("This is most convincing", key=f"rate_{task['task_id']}_{option.option_id}", use_container_width=True):
                payload = {
                    "username": username,
                    "task_id": task["task_id"],
                    "winner_source": option.source,
                    "winner_submission_id": option.submission_id,
                    "option_payload": json.loads(json.dumps([option.__dict__ for option in options])),
                }
                try:
                    upsert_rating(supabase, payload)
                except Exception as exc:
                    st.error("Could not save rating to Supabase.")
                    st.exception(exc)
                    return
                st.rerun()


def render_done() -> None:
    st.header("Thank you for participating!")
    left, right = st.columns(2)
    with left:
        if st.button("Play another round", type="primary"):
            st.session_state.completed_intro = True
            st.session_state.rating_intro_seen = False
            st.session_state.selected_image_id = None
            st.rerun()
    with right:
        st.write("Use the logout button above to end your session.")


if __name__ == "__main__":
    main()
