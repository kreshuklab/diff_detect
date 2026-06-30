from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import streamlit as st
from loguru import logger
from st_supabase_connection import SupabaseConnection
from streamlit_drawable_canvas import st_canvas

# Streamlit runs this file as a script, so expose the src package root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_detect.models import (
    Choice,
    ImageKey,
    RatingEval,
    SelectionChoice,
    SelectionChoiceKey,
)
from diff_detect.storage import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DIFFERENCE_LABEL_STYLES,
    DIFFERENCE_LABELS,
    available_dataset_ids,
    build_rating_options,
    canvas_has_objects,
    canvas_labels,
    choose_rounds,
    completed_task_ids,
    composite_annotation,
    configured_dataset_id,
    decode_png,
    fetch_peer_submission,
    fetch_user_ratings,
    fetch_user_submissions,
    image_for_id,
    label_display,
    load_image,
    load_rounds,
    load_seeded_annotations,
    reference_images,
    upsert_rating_eval,
    upsert_selection_choice,
)

DifferenceLabel = Literal["shape", "color", "texture"]
UserRole = Literal["participant", "maintainer"]
Round = Any
RoundImage = Any
SeededAnnotation = Any

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
                border: 1px solid {DIFFERENCE_LABEL_STYLES[label]["color"]};
                border-radius: 8px;
                padding: 0.38rem 0.7rem;
                background: {DIFFERENCE_LABEL_STYLES[label]["fill"]};
                min-width: 6rem;
                justify-content: center;
            }}
            div.st-key-annotation_tool_selector
                div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child({index}):has(input:checked) {{
                box-shadow: inset 0 0 0 2px {DIFFERENCE_LABEL_STYLES[label]["color"]};
                background: {DIFFERENCE_LABEL_STYLES[label]["fill"]};
            }}
            div.st-key-annotation_tool_selector
                div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-child({index}) p {{
                color: {DIFFERENCE_LABEL_STYLES[label]["color"]};
                font-weight: 700;
            }}
            """
            for index, label in enumerate(DIFFERENCE_LABELS, start=1)
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


def render_label_chips(labels: list[str] | str | None) -> None:
    tokens = parse_label_tokens(labels)
    chips = []
    for token in tokens:
        style = DIFFERENCE_LABEL_STYLES.get(cast(Any, token))
        color = style["color"] if style else "#59636e"
        fill = style["fill"] if style else "rgba(89, 99, 110, 0.14)"
        chips.append(
            f"""
            <span style="
                display: inline-flex;
                align-items: center;
                border: 1px solid {color};
                border-radius: 8px;
                background: {fill};
                color: {color};
                font-weight: 700;
                padding: 0.22rem 0.52rem;
                line-height: 1.2;
            ">{html.escape(token)}</span>
            """
        )

    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.4rem;
            margin: 0.35rem 0 0.45rem;
        ">
            <span style="color: rgba(49, 51, 63, 0.75); font-size: 0.875rem;">
                Label:
            </span>
            {"".join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_label_tokens(labels: list[str] | str | None) -> list[str]:
    if isinstance(labels, list):
        return [label.strip() for label in labels if label.strip()]
    if isinstance(labels, str):
        tokens = [label.strip() for label in labels.split(",")]
        return [label for label in tokens if label]
    return []


def init_state() -> None:
    defaults: dict[str, Any] = {
        "phase": "intro",
        "round_index": 0,
        "dataset_id": None,
        "dataset_selected": False,
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

    dataset_ids = available_dataset_ids()
    dataset_id = current_dataset_id(dataset_ids)
    if dataset_id is None:
        render_dataset_selection(supabase, username, dataset_ids)
        return

    with st.sidebar:
        st.caption(f"Dataset: {dataset_id}")
        if st.button("Change dataset"):
            clear_dataset_selection()
            st.rerun()

    try:
        rounds = choose_rounds(load_rounds(dataset_id), username, dataset_id=dataset_id)
        seeded_annotations = load_seeded_annotations(dataset_id)
    except FileNotFoundError:
        st.error(
            f"Dataset `{dataset_id}` is missing its dataset or challenge JSON file."
        )
        return
    except ValueError as exc:
        st.error(f"Dataset `{dataset_id}` is not valid.")
        st.exception(exc)
        return

    if not rounds:
        st.error(f"Dataset `{dataset_id}` does not contain any rounds.")
        return

    try:
        submissions = fetch_user_submissions(supabase, username, dataset_id)
        ratings = fetch_user_ratings(supabase, username, dataset_id)
    except Exception as exc:
        st.error("Failed to load study progress.")
        st.exception(exc)
        return

    task_ids = {task.task_id for task in rounds}
    submitted_task_ids = completed_task_ids(submissions, task_ids)
    rated_task_ids = completed_task_ids(ratings, task_ids)

    if not st.session_state.completed_intro:
        render_intro(username, len(rounds))
    elif len(submitted_task_ids) < len(rounds):
        render_selection_or_annotation(
            supabase, username, dataset_id, rounds, submissions, debug_mode
        )

    elif not st.session_state.rating_intro_seen:
        render_rating_intro(len(submitted_task_ids))
    elif len(rated_task_ids) < len(rounds):
        render_rating(
            supabase,
            username,
            dataset_id,
            rounds,
            submissions,
            rated_task_ids,
            seeded_annotations,
            debug_mode,
        )
    else:
        render_done()


@dataclass(frozen=True)
class DatasetProgress:
    dataset_id: str
    round_count: int
    submitted_count: int
    rated_count: int
    load_error: str | None = None

    @property
    def is_selectable(self) -> bool:
        return self.load_error is None and self.round_count > 0


def preferred_dataset_id() -> str | None:
    try:
        secret_dataset_id = st.secrets.get("DATASET_ID")
    except Exception:
        secret_dataset_id = None
    configured = str(secret_dataset_id) if secret_dataset_id else None
    return configured_dataset_id(configured)


def current_dataset_id(dataset_ids: list[str]) -> str | None:
    dataset_id = st.session_state.get("dataset_id")
    if (
        st.session_state.get("dataset_selected")
        and isinstance(dataset_id, str)
        and dataset_id in dataset_ids
    ):
        return dataset_id
    clear_dataset_selection()
    return None


def select_dataset(dataset_id: str) -> None:
    reset_dataset_state_if_changed(dataset_id)
    st.session_state.dataset_selected = True


def clear_dataset_selection() -> None:
    st.session_state.dataset_selected = False
    st.session_state.dataset_id = None
    st.session_state.completed_intro = False
    st.session_state.rating_intro_seen = False
    clear_selected_image()


def reset_dataset_state_if_changed(dataset_id: str) -> None:
    previous_dataset_id = st.session_state.get("dataset_id")
    if previous_dataset_id == dataset_id:
        return
    st.session_state.dataset_id = dataset_id
    st.session_state.completed_intro = False
    st.session_state.rating_intro_seen = False
    clear_selected_image()


def load_dataset_progress(
    supabase: SupabaseConnection, username: str, dataset_ids: list[str]
) -> list[DatasetProgress]:
    progress: list[DatasetProgress] = []
    for dataset_id in dataset_ids:
        try:
            rounds = load_rounds(dataset_id)
        except Exception as exc:
            progress.append(
                DatasetProgress(
                    dataset_id=dataset_id,
                    round_count=0,
                    submitted_count=0,
                    rated_count=0,
                    load_error=str(exc),
                )
            )
            continue

        task_ids = {task.task_id for task in rounds}
        submissions = fetch_user_submissions(supabase, username, dataset_id)
        ratings = fetch_user_ratings(supabase, username, dataset_id)
        progress.append(
            DatasetProgress(
                dataset_id=dataset_id,
                round_count=len(rounds),
                submitted_count=len(completed_task_ids(submissions, task_ids)),
                rated_count=len(completed_task_ids(ratings, task_ids)),
            )
        )
    return progress


def render_dataset_selection(
    supabase: SupabaseConnection, username: str, dataset_ids: list[str]
) -> None:
    st.header("Choose a dataset")

    if not dataset_ids:
        st.error(
            "No datasets found. Add a `<dataset_id>.json` file under `data/<dataset_id>/`."
        )
        return

    try:
        progress = load_dataset_progress(supabase, username, dataset_ids)
    except Exception as exc:
        st.error("Failed to load dataset progress.")
        st.exception(exc)
        return

    progress_by_id = {item.dataset_id: item for item in progress}
    st.dataframe(
        [
            {
                "Dataset": item.dataset_id,
                "Rounds": item.round_count,
                "Selections": progress_label(item.submitted_count, item.round_count),
                "Ratings": progress_label(item.rated_count, item.round_count),
                "Status": dataset_status(item),
            }
            for item in progress
        ],
        hide_index=True,
        use_container_width=True,
    )

    selectable_ids = [item.dataset_id for item in progress if item.is_selectable]
    if not selectable_ids:
        st.error("No playable datasets are available.")
        return

    preferred_id = preferred_selectable_dataset_id(selectable_ids)
    selected_id = st.radio(
        "Dataset",
        selectable_ids,
        index=selectable_ids.index(preferred_id),
        format_func=lambda dataset_id: dataset_option_label(progress_by_id[dataset_id]),
    )
    if st.button("Continue", type="primary"):
        select_dataset(selected_id)
        st.rerun()


def preferred_selectable_dataset_id(selectable_ids: list[str]) -> str:
    try:
        preferred_id = preferred_dataset_id()
    except ValueError:
        preferred_id = None
    if preferred_id in selectable_ids:
        return preferred_id
    return selectable_ids[0]


def dataset_option_label(progress: DatasetProgress) -> str:
    return (
        f"{progress.dataset_id} - "
        f"selections {progress_label(progress.submitted_count, progress.round_count)}, "
        f"ratings {progress_label(progress.rated_count, progress.round_count)}"
    )


def progress_label(completed: int, total: int) -> str:
    return f"{completed}/{total}" if total else "0/0"


def dataset_status(progress: DatasetProgress) -> str:
    if progress.load_error:
        return "Unavailable"
    if progress.round_count == 0:
        return "No rounds"
    if progress.submitted_count < progress.round_count:
        return "Selection in progress"
    if progress.rated_count < progress.round_count:
        return "Rating in progress"
    return "Complete"


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
                "Run `schema/supabase.sql` in the Supabase SQL editor."
            )
        else:
            st.error(
                "Login is not configured yet. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in Supabase credentials."
            )
        st.exception(exc)
        return


def render_intro(username: str, round_count: int) -> None:
    st.header(f"Hello {username}!")
    st.write(
        "You will be shown sets of butterfly wings, all but one of which are from the same species. "
        "Please select the one from a different species."
    )
    st.write(
        "Do not choose the odd one out by a damaged wing or any other non-biological difference."
        f"\nAfter you choose, label your selection to explain why you think it is the different species."
        f"\nRepeat this for all {round_count} {pluralize_round(round_count)} in the dataset."
    )
    if st.button("Start", type="primary"):
        st.session_state.completed_intro = True
        st.rerun()


def pluralize_round(round_count: int) -> str:
    return "round" if round_count == 1 else "rounds"


def render_selection_or_annotation(
    supabase: Any,
    username: str,
    dataset_id: str,
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
        render_annotation(supabase, username, dataset_id, next_round, debug_mode)


def render_selection(task: Round, debug_mode: bool) -> None:
    with st.container(key=f"selection_round_{task.task_id}"):
        st.header("Please select the one of a different species than the others.")
        render_debug_task_summary(task, debug_mode)
        shuffled_images = list(task.images)
        columns = st.columns(len(shuffled_images))
        image_slots = [column.empty() for column in columns]
        for slot in image_slots:
            with slot:
                render_image_placeholder(250)

        for index, image_spec in enumerate(shuffled_images):
            with image_slots[index]:
                image = load_image(image_spec, size=(360, 250))
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
    supabase: Any, username: str, dataset_id: str, task: Round, debug_mode: bool
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
                st.image(load_image(reference, size=(220, 140)), width="stretch")
            render_debug_image_info(reference, debug_mode)

    with left:
        render_debug_image_info(selected_spec, debug_mode)
        inject_annotation_tool_styles()
        with st.container(key="annotation_tool_selector"):
            label = cast(
                DifferenceLabel,
                st.radio(
                    "Active difference label",
                    DIFFERENCE_LABELS,
                    format_func=label_display,
                    horizontal=True,
                ),
            )
        style = DIFFERENCE_LABEL_STYLES[label]
        canvas_slot = st.empty()
        with canvas_slot:
            render_image_placeholder(CANVAS_HEIGHT)

        selected_image = load_image(selected_spec)
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
                serialized_canvas_json = json_ready(current_canvas_json)
                annotation_layer = {
                    "mode": "single_canvas_color_coded_labels",
                    "labels": labels,
                    "canvas_json": serialized_canvas_json,
                }
                selection_choice = SelectionChoice(
                    images=selection_image_keys(task, dataset_id),
                    user=username,
                    index=selected_image_index(task, selected_id),
                    explanation=explanation.strip() or None,
                    user_kind="human",
                    annotations=[annotation_layer],
                )
                composite = composite_annotation(
                    selected_image, canvas_result.image_data
                )
                upsert_selection_choice(
                    supabase,
                    selection_choice,
                    dataset_id=dataset_id,
                    task_id=task.task_id,
                    selected_image_id=selected_id,
                    labels=labels,
                    canvas_json=serialized_canvas_json,
                    composite_png_base64=composite,
                )
                clear_selected_image()
                st.session_state.pop(canvas_state_key, None)
                st.rerun()


def selection_image_keys(task: Round, dataset_id: str) -> tuple[ImageKey, ...]:
    return tuple(
        ImageKey(dataset_id=image.dataset_id or dataset_id, image_id=image.image_id)
        for image in task.images
    )


def selected_image_index(task: Round, selected_image_id: str) -> int:
    for index, image in enumerate(task.images):
        if image.image_id == selected_image_id:
            return index
    raise KeyError(f"Image {selected_image_id!r} is not part of task {task.task_id!r}.")


def json_ready(value: Any) -> dict[str, Any]:
    serialized = json.loads(json.dumps(value))
    if not isinstance(serialized, dict):
        raise TypeError("Expected a JSON object.")
    return serialized


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
    dataset_id: str,
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
    peer_submission = fetch_peer_submission(
        supabase, username, task.task_id, dataset_id
    )
    options = build_rating_options(
        task,
        own_submission,
        peer_submission,
        seeded_annotations,
        username,
        dataset_id,
    )

    st.header(
        "For this set of butterfly wings, choose who selected the single species most convincingly."
    )
    render_debug_task_summary(task, debug_mode)
    with st.container(key=f"rating_round_images_{task.task_id}"):
        cols = st.columns(len(task.images))
        image_slots = [column.empty() for column in cols]
        for slot in image_slots:
            with slot:
                render_image_placeholder(180)
        for index, image_spec in enumerate(task.images):
            with image_slots[index]:
                st.image(load_image(image_spec, size=(260, 180)), width="stretch")
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
                render_label_chips(option.label)
                if option.explanation:
                    st.write(option.explanation)
                if st.button(
                    "This is most convincing",
                    key=f"rate_{task.task_id}_{option.option_id}",
                    width="stretch",
                ):
                    rating_eval = RatingEval(
                        user=username,
                        choices=[
                            selection_choice_key_for_option(
                                task, dataset_id, rating_option
                            )
                            for rating_option in options
                        ],
                        most_convincing=Choice(index=index),
                    )
                    try:
                        upsert_rating_eval(
                            supabase,
                            rating_eval,
                            dataset_id=dataset_id,
                            task_id=task.task_id,
                            options=options,
                        )
                    except Exception as exc:
                        st.error("Could not save rating to Supabase.")
                        st.exception(exc)
                        return
                    st.rerun()


def selection_choice_key_for_option(
    task: Round, dataset_id: str, option: Any
) -> SelectionChoiceKey:
    return SelectionChoiceKey(
        images=selection_image_keys(task, dataset_id),
        user=option_user_id(option),
    )


def option_user_id(option: Any) -> str:
    return f"{option.source}:{option.submission_id or option.option_id}"


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
        if st.button("Choose another dataset", type="primary"):
            clear_dataset_selection()
            st.rerun()
    with right:
        st.write("Use the logout button above to end your session.")
