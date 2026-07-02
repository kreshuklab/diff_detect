import json
from typing import Any, Callable, Literal, assert_never, cast

import streamlit as st
from passlib.hash import pbkdf2_sha256
from PIL import Image as PILImage
from streamlit.navigation.page import StreamlitPage
from streamlit_drawable_canvas import st_canvas

from .._state import state
from ..challenges import DATA_DIR
from ..models import (
    ActiveExplainChallenge,
    ActiveRateChallenge,
    Dataset,
    DatasetId,
    ExplainOutcome,
    Image,
    ImageId,
    RateOutcome,
    User,
    UserKind,
    UserRole,
)
from ..storage_sqlite import SqliteStorage

st.set_page_config(page_title="SpeciFly", page_icon=":butterfly:")

PageKey = Literal["login", "challenge", "task", "thanks"]
CHALLENGE_NAMES = {
    "explain_dummy": "Dummy",
    "explain_butterfly_easy": "Butterfly Wings (Easy)",
    "explain_butterfly_difficult": "Butterfly Wings (Difficult)",
    "rate_dummy": "Dummy",
    "rate_butterfly_easy": "Butterfly (Easy)",
    "rate_butterfly_difficult": "Butterfly (Difficult)",
}
DifferenceLabel = Literal["shape", "color", "texture"]
DIFFERENCE_LABEL_STYLES: dict[DifferenceLabel, dict[str, str]] = {
    "shape": {"color": "#ffb000", "fill": "rgba(255, 176, 0, 0.2)"},
    "color": {"color": "#e83e8c", "fill": "rgba(232, 62, 140, 0.18)"},
    "texture": {"color": "#006d77", "fill": "rgba(0, 109, 119, 0.18)"},
}
DIFFERENCE_LABELS: tuple[DifferenceLabel, ...] = tuple(DIFFERENCE_LABEL_STYLES)


def _json_ready(value: Any) -> dict[str, Any]:
    serialized = json.loads(json.dumps(value))
    if not isinstance(serialized, dict):
        raise TypeError("Expected a JSON object.")
    return serialized


def _latest_canvas_widget_json(canvas_widget_key: str) -> dict[str, Any] | None:
    component_value = st.session_state.get(canvas_widget_key)
    if not isinstance(component_value, dict):
        return None
    raw = component_value.get("raw")
    return raw if isinstance(raw, dict) else None


def _canvas_object_dicts(canvas_json: Any | None) -> list[dict[str, Any]]:
    if canvas_json is None:
        return []
    if hasattr(canvas_json, "objects"):
        return [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in canvas_json.objects
        ]
    if not isinstance(canvas_json, dict):
        return []
    objects = canvas_json.get("objects", [])
    return [item for item in objects if isinstance(item, dict)]


def _canvas_labels(
    canvas_json: Any | None, fallback_label: DifferenceLabel | None = None
) -> list[DifferenceLabel]:
    if not canvas_json:
        return [fallback_label] if fallback_label else []

    label_by_color: dict[str, DifferenceLabel] = {
        style["color"].lower(): label
        for label, style in DIFFERENCE_LABEL_STYLES.items()
    }
    labels: list[DifferenceLabel] = []
    for item in _canvas_object_dicts(canvas_json):
        stroke = str(item.get("stroke", "")).lower()
        label = label_by_color.get(stroke)
        if label and label not in labels:
            labels.append(label)

    if not labels and fallback_label and _canvas_object_dicts(canvas_json):
        labels.append(fallback_label)

    return labels


def _annotation_labels(annotations: dict[str, Any] | None) -> list[str]:
    if not annotations:
        return []
    labels = annotations.get("labels")
    if isinstance(labels, list):
        return [str(label) for label in labels if str(label)]
    return []


def _build_annotation_payload(
    canvas_json: Any | None, fallback_label: DifferenceLabel
) -> dict[str, Any] | None:
    if not _canvas_object_dicts(canvas_json):
        return None
    serialized_canvas_json = _json_ready(canvas_json)
    return {
        "mode": "single_canvas_color_coded_labels",
        "labels": _canvas_labels(serialized_canvas_json, fallback_label),
        "canvas_json": serialized_canvas_json,
    }


def _label_display(label: DifferenceLabel) -> str:
    return label.title()


def _render_image_placeholder(height: int) -> None:
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


@st.cache_data(max_entries=100, show_spinner=True)
def _load_study_image(image: Image) -> PILImage.Image:
    path = DATA_DIR / image.source
    assert path.exists(), f"Image file not found: {path}"
    return PILImage.open(path).convert("RGB")


class PageBuilder:
    def __init__(self) -> None:

        self.storage = SqliteStorage()
        if state.toaster:
            # when streamlit reruns toast messages disappear immediately,
            # so we put them in the toaster instead if we intend to rerun and render them here instead
            st.toast(state.toaster)
            del state.toaster

        self.pages: dict[PageKey, StreamlitPage] = {
            "login": st.Page(
                self.render_login_page,
                title="Login",
                icon=":material/login:",
                url_path="login",
                default=state.user is None,
            ),
            "challenge": st.Page(
                self.render_challenge_selection_page,
                title="Select challenge",
                icon=":material/list_alt:",
                url_path="challenge",
                default=state.user is not None,
            ),
            "task": st.Page(
                self.render_task_page,
                title="Current task",
                icon=":material/psychology_alt:",
                url_path="task",
            ),
        }
        if state.user is not None:
            with st.sidebar:
                st.info(f"Logged in as {state.user.id}.")
                st.button(
                    "Logout",
                    on_click=state.reset,
                    type="secondary",
                    width="stretch",
                    icon=":material/logout:",
                )

    def render_login_page(self) -> None:
        st.set_page_config(layout="centered")
        st.title(":butterfly: Welcome to SpeciFly!")
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
                    self.switch_to("challenge")

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
                user = self.storage.fetch_user(typed_user_id)
                if user is None:
                    st.error(
                        f"User '{typed_user_id}' not found. Please create an account."
                    )
                elif pbkdf2_sha256.verify(typed_password, user.hashed_password):
                    state.toaster = f"Welcome back, {user.id}!"

                    state.user = user
                    self.switch_to("challenge")
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
                typed_new_password = None
                retyped_new_password = None

            if st.form_submit_button("Create account"):
                if not typed_new_user_id:
                    st.error("Please enter a username.")
                    return

                user = self.storage.fetch_user(typed_new_user_id)
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
                    kind=UserKind.HUMAN,
                    role=UserRole.PARTICIPANT,
                    hashed_password=pbkdf2_sha256.hash(typed_new_password),
                )
                self.storage.add_user(user)
                st.success(f"Account created for {typed_new_user_id}!")
                state.user = user
                return

    def render_challenge_selection_page(self) -> None:
        st.set_page_config(layout="centered")
        user = self.get_user()
        st.header("Choose a challenge")
        challenge_data = self.storage.fetch_challenges(user)
        if not challenge_data.explain_challenges:
            st.error("No challenges found.")
            st.stop()

        explain_col, rate_col = st.columns(2)
        with explain_col:
            st.subheader("Detect differences")
        with rate_col:
            st.subheader("Rate differences")

        for (
            explain_challenge_id,
            explain_challenge,
        ) in challenge_data.explain_challenges.items():
            explain_col, rate_col = st.columns(2)
            with explain_col:
                if st.button(
                    CHALLENGE_NAMES[explain_challenge_id],
                    key=explain_challenge_id,
                    disabled=explain_challenge.finished,
                ):
                    # challenge = challenge_data.explain_challenges[explain_challenge_id]
                    state.active_challenge = ActiveExplainChallenge(
                        challenge_data=challenge_data, challenge_id=explain_challenge_id
                    )
                    state.task_idx = explain_challenge.first_undone or 0
                    self.switch_to("task")
                st.progress(
                    explain_challenge.progress,
                    text=f"{explain_challenge.done_count}/{explain_challenge.task_count}",
                )

            if explain_challenge_id == "explain_dummy":
                rate_challenge_id = "rate_dummy"
            elif explain_challenge_id == "explain_butterfly_easy":
                rate_challenge_id = "rate_butterfly_easy"
            elif explain_challenge_id == "explain_butterfly_difficult":
                rate_challenge_id = "rate_butterfly_difficult"
            else:
                assert_never(explain_challenge_id)

            rate_challenge = challenge_data.challenges.get(rate_challenge_id)
            with rate_col:
                if (
                    st.button(
                        CHALLENGE_NAMES[rate_challenge_id],
                        key=rate_challenge_id,
                        disabled=rate_challenge is None or rate_challenge.finished,
                    )
                    and rate_challenge is not None
                ):
                    state.active_challenge = ActiveRateChallenge(
                        challenge_data=challenge_data, challenge_id=rate_challenge_id
                    )
                    state.task_idx = rate_challenge.first_undone or 0
                    self.switch_to("task")
                st.progress(
                    0 if rate_challenge is None else rate_challenge.progress,
                    text=f"{0 if rate_challenge is None else rate_challenge.done_count}/{explain_challenge.task_count if rate_challenge is None else rate_challenge.task_count}",
                )

    def render_task_page(self) -> None:
        active = state.active_challenge
        if not active:
            state.toaster = "Please select a challenge first."
            self.switch_to("challenge")

        st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

        if isinstance(active, ActiveExplainChallenge):
            challenge = active.challenge_data.explain_challenges[active.challenge_id]
            save = self._render_explain_task(active)
        elif isinstance(active, ActiveRateChallenge):
            challenge = active.challenge_data.challenges[active.challenge_id]
            save = self._render_rate_task(active)
        else:
            st.error("Challenge state not found.")
            self.switch_to("challenge")

        bottom_left, bottom_center, bottom_right = st.columns(
            [0.1, 0.8, 0.1], vertical_alignment="bottom"
        )
        with bottom_left:
            if state.task_idx > 0:
                if st.button("Previous", width="stretch"):
                    save()
                    state.task_idx -= 1
                    st.rerun()

        with bottom_center:
            task_cols = st.columns(
                [1 for _ in range(0, state.task_idx)]
                + [2]
                + [1 for _ in range(state.task_idx + 1, challenge.task_count)],
                gap="xxsmall",
                vertical_alignment="center",
            )
            for idx, col in enumerate(task_cols):
                with col:
                    if st.button(
                        str(idx + 1) if idx == state.task_idx else "",
                        help=f"Task {idx + 1}",
                        key=f"task_button_{idx}",
                        width="stretch",
                        type="primary"
                        if idx == state.task_idx
                        # else "secondary"
                        # if isinstance(
                        #     challenge.tasks[idx], (ExplainOutcome, RateOutcome)
                        # )
                        else "tertiary",
                        icon=":material/check_box:"
                        if isinstance(
                            challenge.tasks[idx], (ExplainOutcome, RateOutcome)
                        )
                        else ":material/check_box_outline_blank:",
                        icon_position="left",
                    ):
                        save()
                        state.task_idx = idx
                        st.rerun()
            # st.progress(
            #     challenge.progress,
            #     text=f"{CHALLENGE_NAMES[challenge.id]} - {challenge.done_count}/{challenge.task_count}",
            # )
        with bottom_right:
            if state.task_idx < challenge.task_count - 1:
                if st.button("Next", width="stretch"):
                    save()
                    state.task_idx += 1
                    st.rerun()
            else:
                if st.button("Finish", width="stretch"):
                    save()
                    toast = f"Thank you for finishing {CHALLENGE_NAMES[challenge.id]}!"
                    # show toast immediately
                    st.toast(toast)
                    # and again on the challenge page
                    state.toaster = toast
                    st.balloons()
                    # TODO: sleep?
                    self.switch_to("challenge")

    def _render_explain_task(
        self, active: ActiveExplainChallenge
    ) -> Callable[[], None]:
        user = self.get_user()
        explain_task = active.challenge.tasks[state.task_idx]

        annotated_image = self._get_image(
            active.challenge_data.datasets,
            explain_task.dataset_id,
            explain_task.annotated_image,
        )
        reference_specs = [
            self._get_image(
                active.challenge_data.datasets,
                explain_task.dataset_id,
                explain_task.reference_image1,
            ),
            self._get_image(
                active.challenge_data.datasets,
                explain_task.dataset_id,
                explain_task.reference_image2,
            ),
        ]

        left, right = st.columns([2, 1])
        with left:
            subleft, subright = st.columns([1, 1], vertical_alignment="bottom")
            with subleft:
                st.subheader("Annotate visual biological differences")
            with subright:
                label = st.radio(
                    "Active difference label",
                    DIFFERENCE_LABELS,
                    label_visibility="collapsed",
                    format_func=_label_display,
                    horizontal=True,
                    key=f"explain_label_{active.challenge.id}_{state.task_idx}",
                )
        with right:
            st.subheader("References")
            self._render_reference_images(reference_specs)

        canvas_widget_key = (
            f"explain_canvas_{active.challenge.id}_{state.task_idx}_"
            f"{explain_task.annotated_image}"
        )
        canvas_state_key = f"{canvas_widget_key}_json"
        latest_widget_json = _latest_canvas_widget_json(canvas_widget_key)
        if latest_widget_json is not None:
            st.session_state[canvas_state_key] = latest_widget_json

        initial_canvas_json = st.session_state.get(canvas_state_key)
        assert initial_canvas_json is None or isinstance(initial_canvas_json, dict), (
            "Canvas state must be a dict or None."
        )

        with left:
            annotated_canvas_image = _load_study_image(annotated_image)

            if initial_canvas_json is None:
                initial_drawing = None
            else:
                initial_drawing = initial_canvas_json

            original_canvas_width, original_canvas_height = annotated_canvas_image.size
            canvas_scale = 0.23
            canvas_width = round(original_canvas_width * canvas_scale)
            canvas_height = round(original_canvas_height * canvas_scale)
            # canvas_width = 1000
            # canvas_height = 567
            if (
                abs(
                    (original_ratio := original_canvas_width / original_canvas_height)
                    - (canvas_ratio := canvas_width / canvas_height)
                )
                > 0.02
            ):
                st.warning(
                    f"Canvas aspect ratio {original_ratio:.2f} ({original_canvas_width} x {original_canvas_height}) "
                    f"does not match display size {canvas_ratio:.2f} ({canvas_width} x {canvas_height})."
                )

            annotated_canvas_image = annotated_canvas_image.resize(
                (canvas_width, canvas_height), resample=PILImage.Resampling.LANCZOS
            )
            canvas_result = st_canvas(
                fill_color=DIFFERENCE_LABEL_STYLES[label]["fill"],
                stroke_width=8,
                stroke_color=DIFFERENCE_LABEL_STYLES[label]["color"],
                background_image=annotated_canvas_image,  # pyright: ignore[reportArgumentType]
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="freedraw",
                display_toolbar=True,
                key=canvas_widget_key,
                initial_drawing=initial_drawing,  # pyright: ignore[reportArgumentType]
            )
            if canvas_result.json_data is not None:
                st.session_state[canvas_state_key] = canvas_result.json_data
            current_canvas_json = cast(
                dict[str, Any] | None,
                canvas_result.json_data or st.session_state.get(canvas_state_key),
            )

        explanation_key = (
            f"explain_text_{active.challenge.id}_{state.task_idx}_"
            f"{explain_task.annotated_image}"
        )
        explanation = st.text_input(
            "Optional Explanation",
            width="stretch",
            placeholder="Describe the visible difference from the references.",
            key=explanation_key,
            max_chars=244,
        )

        def save() -> None:
            annotation_payload = _build_annotation_payload(current_canvas_json, label)
            cleaned_explanation = explanation.strip()
            if annotation_payload is None and not cleaned_explanation:
                return

            outcome = ExplainOutcome(
                dataset_id=explain_task.dataset_id,
                annotated_image=explain_task.annotated_image,
                reference_image1=explain_task.reference_image1,
                reference_image2=explain_task.reference_image2,
                user=user.id,
                explanation=cleaned_explanation or None,
                annotations=annotation_payload,
            )
            self.storage.upsert_explain_outcome(outcome)

            active.challenge.tasks[state.task_idx] = outcome
            state.active_challenge = active  # update session state with the new outcome
            st.session_state.pop(canvas_state_key, None)
            st.session_state.pop(explanation_key, None)
            state.toaster = "Explanation saved."

        return save

    def _render_rate_task(self, task: ActiveRateChallenge) -> Callable[[], None]:
        st.error("Rate task rendering not implemented yet.")
        return lambda: None

    def _get_image(
        self,
        datasets: dict[DatasetId, Dataset],
        dataset_id: DatasetId,
        image_id: ImageId,
    ) -> Image:
        if dataset_id not in datasets:
            raise KeyError(f"Dataset '{dataset_id}' missing in active task.")

        dataset = datasets[dataset_id]
        image = dataset.images.get(image_id)
        if image is None:
            raise KeyError(
                f"Image '{image_id}' was not found in dataset '{dataset_id}'."
                + f"available images: {list(dataset.images.keys())}"
            )

        return image

    def _render_reference_images(self, images: list[Image]) -> None:
        for image in images:
            st.image(_load_study_image(image), width="stretch")

    def get_user(self) -> User:
        if not state.user:
            self.switch_to("login")

        user = state.user
        if not user:
            st.error("A named account is required for this study.")
            st.stop()

        return user

    def switch_to(self, page_key: PageKey):
        page = self.pages.get(page_key)
        if page is None:
            st.rerun()
        st.switch_page(page)
