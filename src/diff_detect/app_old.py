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

from diff_detect.challenges import CHALLENGE_IDS, _setup_challenge
from diff_detect.models import (
    ChallengeId,
    Choice,
    ImageKey,
    RatingEval,
    SelectionChoice,
    SelectionChoiceKey,
)
from diff_detect.storage import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_DATASET_ID,
    DIFFERENCE_LABEL_STYLES,
    DIFFERENCE_LABELS,
    build_rating_options,
    canvas_has_objects,
    canvas_labels,
    choose_rounds,
    completed_task_ids,
    composite_annotation,
    configured_challenge_id,
    configured_dataset_id,
    decode_png,
    fetch_peer_submission,
    fetch_user_ratings,
    fetch_user_submissions,
    image_for_id,
    label_display,
    load_image,
    load_seeded_annotations,
    reference_images,
    round_view_from_task,
    upsert_rating_eval,
    upsert_selection_choice,
)

DifferenceLabel = Literal["shape", "color", "texture"]
UserRole = Literal["participant", "maintainer"]
Round = Any
RoundImage = Any
SeededAnnotation = Any
PAGE_BY_KEY: dict[str, Any] = {}

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
        "dataset_id": None,
        "challenge_id": None,
        "challenge_selected": False,
        "selected_image_id": None,
        "selected_task_id": None,
        "storage_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    init_state()
    page = st.navigation(build_pages(), position="sidebar", expanded=True)
    page.run()


def build_pages() -> list[Any]:
    global PAGE_BY_KEY
    if not st.session_state.get("authenticated", False):
        PAGE_BY_KEY = {
            "login": st.Page(
                render_login_page,
                title="Login",
                icon=":material/login:",
                url_path="login",
                default=True,
            )
        }
        return list(PAGE_BY_KEY.values())

    PAGE_BY_KEY = {
        "challenge": st.Page(
            render_challenge_page,
            title="Select challenge",
            icon=":material/list_alt:",
            url_path="challenge",
            default=True,
        ),
        "selection": st.Page(
            render_selection_tasks_page,
            title="Selection tasks",
            icon=":material/rule:",
            url_path="selection",
        ),
        "rating": st.Page(
            render_rating_tasks_page,
            title="Rating tasks",
            icon=":material/rate_review:",
            url_path="rating",
        ),
        "thanks": st.Page(
            render_thank_you_page,
            title="Thank you",
            icon=":material/check_circle:",
            url_path="thanks",
        ),
    }
    return list(PAGE_BY_KEY.values())


def switch_to(page_key: str) -> None:
    page = PAGE_BY_KEY.get(page_key)
    if page is None:
        st.rerun()
    st.switch_page(page)


@dataclass(frozen=True)
class ChallengeProgress:
    dataset_id: str
    challenge_id: ChallengeId
    task_count: int
    submitted_count: int
    rated_count: int
    load_error: str | None = None

    @property
    def key(self) -> str:
        return challenge_key(self.dataset_id, self.challenge_id)

    @property
    def is_selectable(self) -> bool:
        return self.load_error is None and self.task_count > 0


@dataclass(frozen=True)
class StudyContext:
    dataset_id: str
    challenge_id: ChallengeId
    rounds: list[Round]
    submissions: list[dict[str, Any]]
    ratings: list[dict[str, Any]]
    submitted_task_ids: set[str]
    rated_task_ids: set[str]
    seeded_annotations: list[SeededAnnotation]


@dataclass(frozen=True)
class CachedChallenge:
    dataset_id: str
    challenge_id: ChallengeId
    rounds: tuple[Round, ...]
    load_error: str | None = None

    @property
    def key(self) -> str:
        return challenge_key(self.dataset_id, self.challenge_id)

    @property
    def task_count(self) -> int:
        return len(self.rounds)


@st.cache_resource(show_spinner=False)
def global_challenges() -> dict[str, CachedChallenge]:
    challenges: dict[str, CachedChallenge] = {}
    for challenge_id in CHALLENGE_IDS:
        try:
            dataset, challenge_model = _setup_challenge(challenge_id)
            image_by_key = {
                (image.dataset_id, image.image_id): image for image in dataset.root
            }
            rounds = tuple(
                round_view_from_task(
                    task,
                    dataset_id=challenge_model.dataset_id,
                    challenge_id=challenge_model.challenge_id,
                    task_index=task_index,
                    image_by_key=image_by_key,
                )
                for task_index, task in enumerate(challenge_model.tasks)
            )
            dataset_id = challenge_model.dataset_id
            load_error = None
        except Exception as exc:
            dataset_id = DEFAULT_DATASET_ID
            rounds = ()
            load_error = str(exc)

        challenge = CachedChallenge(
            dataset_id=dataset_id,
            challenge_id=challenge_id,
            rounds=rounds,
            load_error=load_error,
        )
        challenges[challenge.key] = challenge
    return challenges


def render_login_page() -> None:
    supabase = supabase_connection()
    if st.session_state.get("authenticated", False):
        switch_to("challenge")

    st.title(":butterfly: Welcome to SpeciFly!")
    st.subheader("Can you tell butterfly species apart?")
    st.write("Please create an account or login.")
    configured_login_form(supabase)


def render_challenge_page() -> None:
    supabase, username, _ = authenticated_page_context()
    render_challenge_selection(supabase, username, global_challenges())


def render_selection_tasks_page() -> None:
    supabase, username, debug_mode = authenticated_page_context()
    context = load_current_study_context(supabase, username)
    if context is None:
        return
    if len(context.submitted_task_ids) >= len(context.rounds):
        clear_selected_image()
        switch_to("rating")

    st.header("Selection tasks")
    render_challenge_progress_summary(
        "Selections", len(context.submitted_task_ids), len(context.rounds)
    )
    render_selection_or_annotation(
        supabase,
        username,
        context.dataset_id,
        context.rounds,
        context.submitted_task_ids,
        debug_mode,
    )


def render_rating_tasks_page() -> None:
    supabase, username, debug_mode = authenticated_page_context()
    context = load_current_study_context(supabase, username)
    if context is None:
        return
    if len(context.submitted_task_ids) < len(context.rounds):
        switch_to("selection")
    if len(context.rated_task_ids) >= len(context.rounds):
        switch_to("thanks")

    st.header("Rating tasks")
    render_challenge_progress_summary(
        "Ratings", len(context.rated_task_ids), len(context.rounds)
    )
    render_rating(
        supabase,
        username,
        context.dataset_id,
        context.rounds,
        context.submissions,
        context.rated_task_ids,
        context.seeded_annotations,
        debug_mode,
    )


def render_thank_you_page() -> None:
    supabase, username, _ = authenticated_page_context()
    context = load_current_study_context(supabase, username)
    if context is None:
        return
    if len(context.submitted_task_ids) < len(context.rounds):
        switch_to("selection")
    if len(context.rated_task_ids) < len(context.rounds):
        switch_to("rating")
    render_done()


def authenticated_page_context() -> tuple[SupabaseConnection, str, bool]:
    supabase = supabase_connection()
    username = require_username()
    debug_mode = render_authenticated_chrome(supabase, username)
    return supabase, username, debug_mode


def supabase_connection() -> SupabaseConnection:
    return st.connection(name="supabase", type=SupabaseConnection)


def require_username() -> str:
    if not st.session_state.get("authenticated", False):
        switch_to("login")
    username = st.session_state.get("username")
    if not isinstance(username, str) or not username:
        st.error("A named account is required for this study.")
        st.stop()
    return username


def render_authenticated_chrome(supabase: SupabaseConnection, username: str) -> bool:
    left_col, right_col = st.columns([0.9, 0.1])
    with left_col:
        st.title(":butterfly: SpeciFly")
    with right_col:
        configured_login_form(supabase)

    role = fetch_user_role(supabase, username)
    selection = current_challenge_selection()
    debug_mode = False
    with st.sidebar:
        if selection is not None:
            dataset_id, challenge_id = selection
            st.caption(f"Dataset: {dataset_id}")
            st.caption(f"Challenge: {challenge_id}")
            if st.button("Change challenge"):
                clear_challenge_selection()
                switch_to("challenge")
        if role == "maintainer":
            st.success("Maintainer")
            debug_mode = st.checkbox("Debug Mode", value=False)
    return debug_mode


def load_current_study_context(
    supabase: SupabaseConnection, username: str
) -> StudyContext | None:
    selection = current_challenge_selection()
    if selection is None:
        switch_to("challenge")
        return None
    dataset_id, challenge_id = selection
    challenge = global_challenges().get(challenge_key(dataset_id, challenge_id))
    if challenge is None:
        switch_to("challenge")
        return None
    if challenge.load_error:
        st.error(f"Challenge `{challenge_id}` could not be loaded.")
        st.code(challenge.load_error)
        return None

    rounds = choose_rounds(
        list(challenge.rounds),
        username,
        dataset_id=challenge.key,
    )
    seeded_annotations = load_seeded_annotations(dataset_id)

    if not rounds:
        st.error(f"Challenge `{challenge_id}` does not contain any tasks.")
        return None

    try:
        submissions = fetch_user_submissions(supabase, username, dataset_id)
        ratings = fetch_user_ratings(supabase, username, dataset_id)
    except Exception as exc:
        st.error("Failed to load study progress.")
        st.exception(exc)
        return None

    task_ids = {task.task_id for task in rounds}
    return StudyContext(
        dataset_id=dataset_id,
        challenge_id=challenge_id,
        rounds=rounds,
        submissions=submissions,
        ratings=ratings,
        submitted_task_ids=completed_task_ids(submissions, task_ids),
        rated_task_ids=completed_task_ids(ratings, task_ids),
        seeded_annotations=seeded_annotations,
    )


def preferred_dataset_id() -> str | None:
    try:
        secret_dataset_id = st.secrets.get("DATASET_ID")
    except Exception:
        secret_dataset_id = None
    configured = str(secret_dataset_id) if secret_dataset_id else None
    return configured_dataset_id(configured)


def preferred_challenge_id() -> ChallengeId | None:
    try:
        secret_challenge_id = st.secrets.get("CHALLENGE_ID")
    except Exception:
        secret_challenge_id = None
    configured = str(secret_challenge_id) if secret_challenge_id else None
    try:
        return configured_challenge_id(configured)
    except ValueError:
        return None


def current_challenge_selection() -> tuple[str, ChallengeId] | None:
    dataset_id = st.session_state.get("dataset_id")
    challenge_id = st.session_state.get("challenge_id")
    if (
        st.session_state.get("challenge_selected")
        and isinstance(dataset_id, str)
        and isinstance(challenge_id, str)
        and challenge_key(dataset_id, challenge_id) in global_challenges()
    ):
        return dataset_id, cast(ChallengeId, challenge_id)
    clear_challenge_selection()
    return None


def select_challenge(dataset_id: str, challenge_id: ChallengeId) -> None:
    reset_challenge_state_if_changed(dataset_id, challenge_id)
    st.session_state.challenge_selected = True


def clear_challenge_selection() -> None:
    st.session_state.challenge_selected = False
    st.session_state.dataset_id = None
    st.session_state.challenge_id = None
    clear_selected_image()


def reset_challenge_state_if_changed(
    dataset_id: str, challenge_id: ChallengeId
) -> None:
    previous_dataset_id = st.session_state.get("dataset_id")
    previous_challenge_id = st.session_state.get("challenge_id")
    if previous_dataset_id == dataset_id and previous_challenge_id == challenge_id:
        return
    st.session_state.dataset_id = dataset_id
    st.session_state.challenge_id = challenge_id
    clear_selected_image()


def load_challenge_progress(
    supabase: SupabaseConnection,
    username: str,
    challenges: dict[ChallengeId, CachedChallenge],
) -> list[ChallengeProgress]:
    progress: list[ChallengeProgress] = []
    dataset_ids = sorted({challenge.dataset_id for challenge in challenges.values()})
    for challenge_id in dataset_ids:
        submissions = fetch_user_submissions(supabase, username, dataset_id)
        ratings = fetch_user_ratings(supabase, username, dataset_id)
        dataset_challenges = sorted(
            (
                challenge
                for challenge in challenges.values()
                if challenge.dataset_id == dataset_id
            ),
            key=lambda challenge: challenge.challenge_id,
        )
        for challenge in dataset_challenges:
            challenge_id = challenge.challenge_id
            if challenge.load_error:
                progress.append(
                    ChallengeProgress(
                        dataset_id=dataset_id,
                        challenge_id=challenge_id,
                        task_count=0,
                        submitted_count=0,
                        rated_count=0,
                        load_error=challenge.load_error,
                    )
                )
                continue

            task_ids = {task.task_id for task in challenge.rounds}
            progress.append(
                ChallengeProgress(
                    dataset_id=dataset_id,
                    challenge_id=challenge_id,
                    task_count=challenge.task_count,
                    submitted_count=len(completed_task_ids(submissions, task_ids)),
                    rated_count=len(completed_task_ids(ratings, task_ids)),
                )
            )
    return progress


def render_challenge_selection(
    supabase: SupabaseConnection,
    username: str,
    challenges: dict[str, CachedChallenge],
) -> None:
    st.header("Choose a challenge")

    if not challenges:
        st.error(
            "No challenges found. Check the source index CSV files under `data/butterfly/download/`."
        )
        return

    try:
        progress = load_challenge_progress(supabase, username, challenges)
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
        select_challenge(selected_progress.dataset_id, selected_progress.challenge_id)
        switch_to(next_page_for_progress(selected_progress))


def preferred_selectable_challenge_key(selectable_keys: list[str]) -> str:
    try:
        dataset_id = preferred_dataset_id()
    except ValueError:
        dataset_id = None
    challenge_id = preferred_challenge_id()
    if dataset_id and challenge_id:
        preferred_key = challenge_key(dataset_id, challenge_id)
        if preferred_key in selectable_keys:
            return preferred_key
    return selectable_keys[0]


def challenge_key(dataset_id: str, challenge_id: str) -> str:
    return f"{dataset_id}/{challenge_id}"


def challenge_option_label(progress: ChallengeProgress) -> str:
    return (
        f"{progress.dataset_id} / {progress.challenge_id} - "
        f"selections {progress_label(progress.submitted_count, progress.task_count)}, "
        f"ratings {progress_label(progress.rated_count, progress.task_count)}"
    )


def progress_label(completed: int, total: int) -> str:
    return f"{completed}/{total}" if total else "0/0"


def challenge_status(progress: ChallengeProgress) -> str:
    if progress.load_error:
        return "Unavailable"
    if progress.task_count == 0:
        return "No tasks"
    if progress.submitted_count < progress.task_count:
        return "Selection in progress"
    if progress.rated_count < progress.task_count:
        return "Rating in progress"
    return "Complete"


def next_page_for_progress(progress: ChallengeProgress) -> str:
    if progress.submitted_count < progress.task_count:
        return "selection"
    if progress.rated_count < progress.task_count:
        return "rating"
    return "thanks"


def render_challenge_progress_summary(label: str, completed: int, total: int) -> None:
    st.caption(f"{label}: {progress_label(completed, total)}")
    if total:
        st.progress(completed / total)


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


def render_selection_or_annotation(
    supabase: Any,
    username: str,
    dataset_id: str,
    rounds: list[Round],
    submitted_task_ids: set[str],
    debug_mode: bool,
) -> None:
    next_round = next(task for task in rounds if task.task_id not in submitted_task_ids)
    progress_index = len(submitted_task_ids) + 1

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
        if st.button("Back to challenge selection", type="primary"):
            clear_challenge_selection()
            switch_to("challenge")
    with right:
        st.write("Use the logout button above to end your session.")
