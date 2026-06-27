from __future__ import annotations

import json
from typing import Any, Literal, cast

import streamlit as st
from loguru import logger
from st_supabase_connection import SupabaseConnection
from streamlit_drawable_canvas import st_canvas

from models import (
    AnnotationLayers,
    DifferenceLabel,
    RatingPayload,
    Round,
    RoundImage,
    SeededAnnotation,
    SubmissionPayload,
    UserRole,
)
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
    st.error(
        "Missing dependency: `st-login-form`. Install dependencies with `pip install -r requirements.txt`."
    )
    raise


st.set_page_config(page_title="SpeciFly", page_icon=":butterfly:", layout="wide")


def inject_annotation_tool_styles() -> None:
    label_rules = "\n".join(
        [
            f"""
            div.st-key-annotation_tool_selector
                div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child({index}) {{
                border: 1px solid {LABEL_STYLES[label]["color"]};
                border-radius: 8px;
                padding: 0.38rem 0.7rem;
                background: {LABEL_STYLES[label]["fill"]};
                min-width: 6rem;
                justify-content: center;
            }}
            div.st-key-annotation_tool_selector
                div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child({index}):has(input:checked) {{
                box-shadow: inset 0 0 0 2px {LABEL_STYLES[label]["color"]};
                background: {LABEL_STYLES[label]["fill"]};
            }}
            div.st-key-annotation_tool_selector
                div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child({index}) p {{
                color: {LABEL_STYLES[label]["color"]};
                font-weight: 700;
            }}
            """
            for index, label in enumerate(LABELS, start=1)
        ]
    )
    st.markdown(
        f"""
        <style>
            div.st-key-annotation_tool_selector
                div[data-testid="stRadio"] div[role="radiogroup"] {{
                gap: 0.55rem;
            }}
            div.st-key-annotation_tool_selector
                div[data-testid="stRadio"] div[role="radiogroup"] > label {{
                margin: 0;
            }}
            {label_rules}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_image_placeholder(height: int) -> None:
    st.markdown(
        f"""
        <div style="
            height: {height}px;
            width: 100%;
            border: 1px solid rgba(49, 51, 63, 0.16);
            background: rgba(250, 250, 250, 0.92);
        "></div>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults: dict[str, Any] = {
        "phase": "intro",
        "round_index": 0,
        "selected_image_id": None,
        "selected_task_id": None,
        "completed_intro": False,
        "rating_intro_seen": False,
        "storage_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    init_state()  # TODO: do we need this?
    left_col, right_col = st.columns([0.9, 0.1])
    supabase = st.connection(name="supabase", type=SupabaseConnection)

    authenticated = st.session_state.get("authenticated", False)

    if not authenticated:
        st.title(":butterfly: Welcome to SpeciFly!")
        st.subheader("Can you tell butterfly species apart?")
        st.write("Please create an account or login.")
        configured_login_form(supabase)
        return

    with left_col:
        st.title(":butterfly: SpeciFly")

    with right_col:
        configured_login_form(supabase)

    username = st.session_state.get("username")
    if not username:
        st.error("A named account is required for this study.")
        return

    role = fetch_user_role(supabase, username)

    if role == "maintainer":
        with st.sidebar:
            st.success("Maintainer")
            debug_mode = st.checkbox("Debug Mode", value=False)
    else:
        debug_mode = False

    rounds = choose_rounds(load_rounds(), username, MIN_ROUNDS)
    seeded_annotations = load_seeded_annotations()

    try:
        submissions = fetch_user_submissions(supabase, username)
        ratings = fetch_user_ratings(supabase, username)
    except Exception as exc:
        st.error("Failed to load study progress.")
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


def fetch_user_role(supabase, username: str) -> Literal["maintainer", "participant"]:
    response = (
        supabase.table("users")
        .select("role")
        .eq("username", username)
        .single()
        .execute()
    )
    role = response.data.get("role", "participant")
    if role not in ("maintainer", "participant"):
        logger.warning(
            f"Unexpected role '{role}' for user '{username}', defaulting to 'participant'."
        )
        role = "participant"

    return cast(UserRole, role)


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
                "Apply `supabase/migrations/20260626193000_create_study_tables.sql` or run `schema/supabase.sql` in the Supabase SQL editor."
            )
        else:
            st.error(
                "Login is not configured yet. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in Supabase credentials."
            )
        st.exception(exc)
        return


def render_intro(username: str) -> None:
    st.header(f"Hello {username}!")
    st.write(
        "You will be shown four sets of butterfly wings, three of which are from the same species. "
        "Please select the one from a different species."
    )
    st.write(
        "Do not choose the odd one out by a damaged wing or any other non-biological difference."
        f"\nAfter you choose, label your selection to explain why you think it is the different species."
        f"\nRepeat this for at least {MIN_ROUNDS} rounds."
    )
    if st.button("Start", type="primary"):
        st.session_state.completed_intro = True
        st.rerun()


def render_selection_or_annotation(
    supabase: Any,
    username: str,
    rounds: list[Round],
    submissions: list[dict[str, Any]],
    debug_mode: bool,
) -> None:
    submitted = {submission["task_id"] for submission in submissions}
    next_round = next(task for task in rounds if task.task_id not in submitted)
    progress_index = len(submitted) + 1

    st.caption(f"Round {progress_index} of {len(rounds)}")
    if not selected_image_belongs_to_task(next_round):
        clear_selected_image()
        render_selection(next_round, debug_mode)
    else:
        render_annotation(supabase, username, next_round, debug_mode)


def render_selection(task: Round, debug_mode: bool) -> None:
    with st.container(key=f"selection_round_{task.task_id}"):
        st.header("Please select the one of a different species than the other three.")
        render_debug_task_summary(task, debug_mode)
        shuffled_images = list(task.images)
        columns = st.columns(4)
        image_slots = [column.empty() for column in columns]
        for slot in image_slots:
            with slot:
                render_image_placeholder(250)

        for index, image_spec in enumerate(shuffled_images):
            with image_slots[index]:
                image = load_wing_image(image_spec, size=(360, 250))
                st.image(image, width="stretch")
            with columns[index]:
                render_debug_image_info(image_spec, debug_mode)
                if st.button(
                    "Select",
                    key=f"select_{task.task_id}_{image_spec.image_id}",
                    width="stretch",
                ):
                    st.session_state.selected_image_id = image_spec.image_id
                    st.session_state.selected_task_id = task.task_id
                    st.rerun()


def render_annotation(
    supabase: Any, username: str, task: Round, debug_mode: bool
) -> None:
    if not selected_image_belongs_to_task(task):
        clear_selected_image()
        st.warning("That selection belonged to another round. Please choose again.")
        st.rerun()

    selected_id = cast(str, st.session_state.selected_image_id)
    selected_spec = image_for_id(task, selected_id)
    references = reference_images(task, selected_id)
    canvas_widget_key = f"canvas_{task.task_id}_{selected_id}"
    canvas_state_key = f"canvas_json_{task.task_id}_{selected_id}"
    latest_widget_json = latest_canvas_widget_json(canvas_widget_key)
    if latest_widget_json is not None:
        st.session_state[canvas_state_key] = latest_widget_json
    initial_canvas_json = st.session_state.get(canvas_state_key)

    st.header("Annotate the differences in the selected one.")
    render_debug_task_summary(task, debug_mode)
    left, right = st.columns([3, 1])

    with right, st.container(key=f"annotation_refs_{task.task_id}_{selected_id}"):
        st.subheader("References")
        reference_slots = [st.empty() for _ in references]
        for slot in reference_slots:
            with slot:
                render_image_placeholder(140)
        for slot, reference in zip(reference_slots, references):
            with slot:
                st.image(load_wing_image(reference, size=(220, 140)), width="stretch")
            render_debug_image_info(reference, debug_mode)

    with left:
        render_debug_image_info(selected_spec, debug_mode)
        inject_annotation_tool_styles()
        with st.container(key="annotation_tool_selector"):
            label = cast(
                DifferenceLabel,
                st.radio(
                    "Active difference label",
                    LABELS,
                    format_func=label_display,
                    horizontal=True,
                ),
            )
        style = LABEL_STYLES[label]
        canvas_slot = st.empty()
        with canvas_slot:
            render_image_placeholder(CANVAS_HEIGHT)

        selected_image = load_wing_image(selected_spec)
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
            "key": canvas_widget_key,
        }
        if initial_canvas_json is not None:
            canvas_kwargs["initial_drawing"] = cast(Any, initial_canvas_json)
        with canvas_slot:
            canvas_result = st_canvas(**canvas_kwargs)
        if canvas_result.json_data is not None:
            st.session_state[canvas_state_key] = canvas_result.json_data
        current_canvas_json = cast(
            dict[str, Any] | None,
            canvas_result.json_data or st.session_state.get(canvas_state_key),
        )
        explanation = st.text_area(
            "Optional explanation",
            placeholder="Briefly describe the visible biological difference.",
        )

        back_col, save_col = st.columns([1, 2])
        with back_col:
            if st.button("Choose another image"):
                clear_selected_image()
                st.session_state.pop(canvas_state_key, None)
                st.rerun()
        with save_col:
            if st.button("Next", type="primary", width="stretch"):
                if current_canvas_json is None or not canvas_has_objects(
                    current_canvas_json
                ):
                    st.warning(
                        "Please add at least one annotation stroke before continuing."
                    )
                    return

                labels = canvas_labels(current_canvas_json, label)
                composite = composite_annotation(
                    selected_image, canvas_result.image_data
                )
                payload = SubmissionPayload(
                    username=username,
                    task_id=task.task_id,
                    selected_image_id=selected_id,
                    label=labels[0],
                    labels=labels,
                    explanation=explanation.strip() or None,
                    canvas_json=current_canvas_json,
                    annotation_layers=AnnotationLayers(
                        mode="single_canvas_color_coded_labels",
                        labels=labels,
                        canvas_json=current_canvas_json,
                    ),
                    composite_png_base64=composite,
                )
                upsert_submission(supabase, payload)
                clear_selected_image()
                st.session_state.pop(canvas_state_key, None)
                st.rerun()


def selected_image_belongs_to_task(task: Round) -> bool:
    selected_id = st.session_state.get("selected_image_id")
    selected_task_id = st.session_state.get("selected_task_id")
    if not isinstance(selected_id, str) or selected_task_id != task.task_id:
        return False
    return any(image.image_id == selected_id for image in task.images)


def clear_selected_image() -> None:
    st.session_state.selected_image_id = None
    st.session_state.selected_task_id = None


def latest_canvas_widget_json(canvas_widget_key: str) -> dict[str, Any] | None:
    component_value = st.session_state.get(canvas_widget_key)
    if not isinstance(component_value, dict):
        return None
    raw = component_value.get("raw")
    return raw if isinstance(raw, dict) else None


def render_rating_intro(completed_count: int) -> None:
    st.header(f"Thank you for explaining {completed_count} selections!")
    st.write("Now, please take some time to rate others' choices.")
    if st.button("Next", type="primary"):
        st.session_state.rating_intro_seen = True
        st.rerun()


def render_rating(
    supabase: Any,
    username: str,
    rounds: list[Round],
    submissions: list[dict[str, Any]],
    rated_task_ids: set[str],
    seeded_annotations: list[SeededAnnotation],
    debug_mode: bool,
) -> None:
    submissions_by_task = {
        submission["task_id"]: submission for submission in submissions
    }
    task = next(task for task in rounds if task.task_id not in rated_task_ids)
    own_submission = submissions_by_task[task.task_id]
    peer_submission = fetch_peer_submission(supabase, username, task.task_id)
    options = build_rating_options(
        task, own_submission, peer_submission, seeded_annotations, username
    )

    st.header(
        "For this set of butterfly wings, choose who selected the single species most convincingly."
    )
    render_debug_task_summary(task, debug_mode)
    with st.container(key=f"rating_round_images_{task.task_id}"):
        cols = st.columns(4)
        image_slots = [column.empty() for column in cols]
        for slot in image_slots:
            with slot:
                render_image_placeholder(180)
        for index, image_spec in enumerate(task.images):
            with image_slots[index]:
                st.image(load_wing_image(image_spec, size=(260, 180)), width="stretch")
            with cols[index]:
                render_debug_image_info(image_spec, debug_mode)

    st.divider()
    with st.container(key=f"rating_options_{task.task_id}"):
        option_cols = st.columns(len(options))
        option_slots = [column.empty() for column in option_cols]
        for slot in option_slots:
            with slot:
                render_image_placeholder(220)
        for index, option in enumerate(options):
            with option_slots[index]:
                st.image(decode_png(option.composite_png_base64), width="stretch")
            with option_cols[index]:
                st.caption(f"Label: {label_display(option.label)}")
                if option.explanation:
                    st.write(option.explanation)
                if st.button(
                    "This is most convincing",
                    key=f"rate_{task.task_id}_{option.option_id}",
                    width="stretch",
                ):
                    payload = RatingPayload(
                        username=username,
                        task_id=task.task_id,
                        winner_source=option.source,
                        winner_submission_id=option.submission_id,
                        option_payload=json.loads(
                            json.dumps(
                                [option.model_dump(mode="json") for option in options]
                            )
                        ),
                    )
                    try:
                        upsert_rating(supabase, payload)
                    except Exception as exc:
                        st.error("Could not save rating to Supabase.")
                        st.exception(exc)
                        return
                    st.rerun()


def render_debug_task_summary(task: Round, debug_mode: bool) -> None:
    if not debug_mode:
        return
    st.info(
        " | ".join(
            [
                f"task: {task.task_id}",
                f"rule: {task.metadata.round_rule}",
                f"mimic group: {task.metadata.mimic_group}",
                f"view: {task.metadata.view}",
            ]
        )
    )


def render_debug_image_info(image_spec: RoundImage, debug_mode: bool) -> None:
    if not debug_mode:
        return
    st.caption(
        " | ".join(
            [
                f"role: {image_spec.species_role}",
                f"species: {image_spec.species}",
                f"subspecies: {image_spec.subspecies}",
                f"view: {image_spec.view}",
                f"mimic: {image_spec.mimic_group}",
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
            clear_selected_image()
            st.rerun()
    with right:
        st.write("Use the logout button above to end your session.")


if __name__ == "__main__":
    main()
