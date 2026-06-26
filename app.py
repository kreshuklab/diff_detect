from __future__ import annotations

import json
from typing import Any, cast

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from study_storage import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    LABEL_STYLES,
    LABELS,
    MIN_ROUNDS,
    build_rating_options,
    canvas_has_objects,
    canvas_labels,
    choose_rounds,
    composite_annotation,
    decode_png,
    fetch_peer_submission,
    fetch_user_ratings,
    fetch_user_submissions,
    image_for_id,
    label_display,
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


st.set_page_config(
    page_title="Butterfly Wing Study", page_icon=":butterfly:", layout="wide"
)


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
    debug_mode = st.sidebar.checkbox("Debug taxonomy", value=False)

    if login_form is None:
        st.error(
            "Missing dependency: `st-login-form`. Install dependencies with `pip install -r requirements.txt`."
        )
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
        message = str(exc)
        if "public.users" in message or "PGRST205" in message:
            st.error(
                "Supabase is connected, but the `public.users` table is missing. "
                "Apply `supabase/migrations/20260626193000_create_study_tables.sql` or run `schema/supabase.sql` in the Supabase SQL editor."
            )
        else:
            st.error(
                "Login is not configured yet. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in Supabase credentials."
            )
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
        st.error(
            "Could not load study progress from Supabase. Check that `schema/supabase.sql` has been applied."
        )
        st.exception(exc)
        return

    submitted_task_ids = {submission["task_id"] for submission in submissions}
    rated_task_ids = {rating["task_id"] for rating in ratings}

    if not st.session_state.completed_intro:
        render_intro(username)
    elif len(submitted_task_ids) < len(rounds):
        render_selection_or_annotation(
            supabase, username, rounds, submissions, debug_mode
        )
    elif not st.session_state.rating_intro_seen:
        render_rating_intro(len(submissions))
    elif len(rated_task_ids) < len(rounds):
        render_rating(
            supabase,
            username,
            rounds,
            submissions,
            rated_task_ids,
            seeded_annotations,
            debug_mode,
        )
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
    debug_mode: bool,
) -> None:
    submitted = {submission["task_id"] for submission in submissions}
    next_round = next(task for task in rounds if task["task_id"] not in submitted)
    progress_index = len(submitted) + 1

    st.caption(f"Round {progress_index} of {len(rounds)}")
    if not st.session_state.selected_image_id:
        render_selection(next_round, debug_mode)
    else:
        render_annotation(supabase, username, next_round, debug_mode)


def render_selection(task: dict[str, Any], debug_mode: bool) -> None:
    st.header("Please select the one of a different species than the other three.")
    render_debug_task_summary(task, debug_mode)
    shuffled_images = list(task["images"])
    columns = st.columns(4)
    for index, image_spec in enumerate(shuffled_images):
        with columns[index]:
            image = load_wing_image(image_spec, size=(360, 250))
            st.image(image, width="stretch")
            render_debug_image_info(image_spec, debug_mode)
            if st.button(
                "Select",
                key=f"select_{task['task_id']}_{image_spec['image_id']}",
                width="stretch",
            ):
                st.session_state.selected_image_id = image_spec["image_id"]
                st.rerun()


def render_annotation(
    supabase: Any, username: str, task: dict[str, Any], debug_mode: bool
) -> None:
    selected_id = st.session_state.selected_image_id
    selected_spec = image_for_id(task, selected_id)
    selected_image = load_wing_image(selected_spec)
    references = reference_images(task, selected_id)
    canvas_state_key = f"canvas_json_{task['task_id']}_{selected_id}"

    st.header("Annotate the differences in the selected one.")
    render_debug_task_summary(task, debug_mode)
    left, right = st.columns([3, 1])

    with right:
        st.subheader("References")
        for reference in references:
            st.image(load_wing_image(reference, size=(220, 140)), width="stretch")
            render_debug_image_info(reference, debug_mode)

    with left:
        render_debug_image_info(selected_spec, debug_mode)
        label = st.radio("Active difference label", LABELS, horizontal=True)
        style = LABEL_STYLES[label]
        canvas_kwargs = {
            "fill_color": style["fill"],
            "stroke_width": 8,
            "stroke_color": style["color"],
            "background_image": cast(Any, selected_image.convert("RGB")),
            "update_streamlit": True,
            "height": CANVAS_HEIGHT,
            "width": CANVAS_WIDTH,
            "drawing_mode": "freedraw",
            "display_toolbar": True,
            "key": f"canvas_{task['task_id']}_{selected_id}",
        }
        if st.session_state.get(canvas_state_key) is not None:
            canvas_kwargs["initial_drawing"] = cast(Any, st.session_state[canvas_state_key])
        canvas_result = st_canvas(**canvas_kwargs)
        if canvas_result.json_data is not None:
            st.session_state[canvas_state_key] = canvas_result.json_data
        explanation = st.text_area(
            "Optional explanation",
            placeholder="Briefly describe the visible biological difference.",
        )

        back_col, save_col = st.columns([1, 2])
        with back_col:
            if st.button("Choose another image"):
                st.session_state.selected_image_id = None
                st.session_state.pop(canvas_state_key, None)
                st.rerun()
        with save_col:
            if st.button("Next", type="primary", width="stretch"):
                if not canvas_has_objects(canvas_result.json_data):
                    st.warning(
                        "Please add at least one annotation stroke before continuing."
                    )
                    return

                labels = canvas_labels(canvas_result.json_data, label)
                composite = composite_annotation(
                    selected_image, canvas_result.image_data
                )
                payload = {
                    "username": username,
                    "task_id": task["task_id"],
                    "selected_image_id": selected_id,
                    "label": labels[0],
                    "labels": labels,
                    "explanation": explanation.strip() or None,
                    "canvas_json": canvas_result.json_data,
                    "annotation_layers": {
                        "mode": "single_canvas_color_coded_labels",
                        "labels": labels,
                        "canvas_json": canvas_result.json_data,
                    },
                    "composite_png_base64": composite,
                }
                upsert_submission(supabase, payload)
                st.session_state.selected_image_id = None
                st.session_state.pop(canvas_state_key, None)
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
    debug_mode: bool,
) -> None:
    submissions_by_task = {
        submission["task_id"]: submission for submission in submissions
    }
    task = next(task for task in rounds if task["task_id"] not in rated_task_ids)
    own_submission = submissions_by_task[task["task_id"]]
    peer_submission = fetch_peer_submission(supabase, username, task["task_id"])
    options = build_rating_options(
        task, own_submission, peer_submission, seeded_annotations, username
    )

    st.header(
        "For this set of butterfly wings, choose who selected the single species most convincingly."
    )
    render_debug_task_summary(task, debug_mode)
    cols = st.columns(4)
    for index, image_spec in enumerate(task["images"]):
        with cols[index]:
            st.image(load_wing_image(image_spec, size=(260, 180)), width="stretch")
            render_debug_image_info(image_spec, debug_mode)

    st.divider()
    option_cols = st.columns(len(options))
    for index, option in enumerate(options):
        with option_cols[index]:
            st.image(decode_png(option.composite_png_base64), width="stretch")
            st.caption(f"Label: {label_display(option.label)}")
            if option.explanation:
                st.write(option.explanation)
            if st.button(
                "This is most convincing",
                key=f"rate_{task['task_id']}_{option.option_id}",
                width="stretch",
            ):
                payload = {
                    "username": username,
                    "task_id": task["task_id"],
                    "winner_source": option.source,
                    "winner_submission_id": option.submission_id,
                    "option_payload": json.loads(
                        json.dumps([option.__dict__ for option in options])
                    ),
                }
                try:
                    upsert_rating(supabase, payload)
                except Exception as exc:
                    st.error("Could not save rating to Supabase.")
                    st.exception(exc)
                    return
                st.rerun()


def render_debug_task_summary(task: dict[str, Any], debug_mode: bool) -> None:
    if not debug_mode:
        return
    metadata = task.get("metadata", {})
    st.info(
        " | ".join(
            [
                f"task: {task.get('task_id', '')}",
                f"rule: {metadata.get('round_rule', '')}",
                f"mimic group: {metadata.get('mimic_group', '')}",
                f"view: {metadata.get('view', '')}",
            ]
        )
    )


def render_debug_image_info(image_spec: dict[str, Any], debug_mode: bool) -> None:
    if not debug_mode:
        return
    st.caption(
        " | ".join(
            [
                f"role: {image_spec.get('species_role', '')}",
                f"species: {image_spec.get('species', '')}",
                f"subspecies: {image_spec.get('subspecies', '')}",
                f"view: {image_spec.get('view', '')}",
                f"mimic: {image_spec.get('mimic_group', '')}",
            ]
        )
    )


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
