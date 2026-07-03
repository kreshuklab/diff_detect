from typing import Any, Callable

import streamlit as st
from PIL import Image as PILImage
from pydantic import ValidationError
from streamlit_drawable_canvas import st_canvas

from diff_detect._patch_canvas_toolbar import patch_canvas_toolbar
from diff_detect.common import (
    CHALLENGE_NAMES,
    DIFFERENCE_LABEL_STYLES,
    DIFFERENCE_LABELS,
)

from ._page_utils import PageKey, format_label
from ._state import state
from ._storage import storage
from ._utils import get_image
from .challenges import load_study_image
from .models import (
    ActiveExplainChallenge,
    ActiveRateChallenge,
    Annotation,
    ExplainOutcome,
    RateOutcome,
    User,
)


def render_task_page() -> PageKey | None:
    user = state.user
    if user is None:
        return "login"

    # remove some padding from the top and sides of the page to make more room for the canvas
    st.markdown(
        """
    <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 1rem;
                padding-left: 4rem;
                padding-right: 4rem;
            }
    </style>
    """,
        unsafe_allow_html=True,
    )

    active = state.active_challenge
    if not active:
        state.toaster = "Please select a challenge first."
        return "challenge"

    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")

    if isinstance(active, ActiveExplainChallenge):
        challenge = active.challenge_data.explain_challenges[active.challenge_id]
        save = _render_explain_task(user, active)
    elif isinstance(active, ActiveRateChallenge):
        challenge = active.challenge_data.challenges[active.challenge_id]
        save = _render_rate_task(user, active)
    else:
        st.error("Challenge state not found.")
        return "challenge"

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
                    if isinstance(challenge.tasks[idx], (ExplainOutcome, RateOutcome))
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
                return "challenge"


def _render_explain_task(
    user: User, active: ActiveExplainChallenge
) -> Callable[[], None]:
    explain_task = active.challenge.tasks[state.task_idx]

    annotated_image = get_image(
        active.challenge_data.datasets,
        explain_task.dataset_id,
        explain_task.annotated_image,
    )
    reference_images = [
        get_image(
            active.challenge_data.datasets,
            explain_task.dataset_id,
            explain_task.reference_image1,
        ),
        get_image(
            active.challenge_data.datasets,
            explain_task.dataset_id,
            explain_task.reference_image2,
        ),
    ]

    canvas_state_base = (
        f"explain_canvas_{active.challenge.id}_{state.task_idx}_"
        f"{explain_task.annotated_image}_state"
    )
    canvas_reset_key = f"{canvas_state_base}_reset_generation"
    canvas_generation = st.session_state.get(canvas_reset_key, 0)
    canvas_state = (
        canvas_state_base
        if canvas_generation == 0
        else f"{canvas_state_base}_{canvas_generation}"
    )

    st.header(CHALLENGE_NAMES[active.challenge.id])
    left, right = st.columns([2, 1])
    with left, st.container(width="content", horizontal=True):
        # header_col, label_radio_col, reset_col = st.columns(
        #     [0.5, 0.3, 0.2], vertical_alignment="bottom"
        # )
        # with header_col:
        st.subheader("Annotate visual biological differences")
        # with label_radio_col:
        label_radio_key = f"explain_label_{active.challenge.id}_{state.task_idx}"

        label = st.radio(
            "Active difference label",
            DIFFERENCE_LABELS,
            label_visibility="collapsed",
            format_func=format_label,
            horizontal=True,
            key=label_radio_key,
        )
        # with reset_col:
        if st.button(
            "Reset labels",
            help="Clear drawn and restored labels",
            icon=":material/delete:",
            key=f"reset_labels_{active.challenge.id}_{state.task_idx}",
            width="content",
        ):
            if isinstance(explain_task, ExplainOutcome):
                if explain_task.explanation:
                    explain_task = explain_task.model_copy(update={"annotation": None})
                    storage.upsert_explain_outcome(explain_task)
                else:
                    storage.delete_explain_outcome(explain_task)
                    explain_task = explain_task.as_explain_task()

                active.challenge.tasks[state.task_idx] = explain_task
                state.active_challenge = active

            st.session_state.pop(canvas_state, None)
            st.session_state[canvas_reset_key] = canvas_generation + 1
            st.rerun()

    with right:
        st.subheader("References")

    left, right = st.columns([2, 1])
    with right:
        for ref_img in reference_images:
            st.image(load_study_image(ref_img), width="stretch")

    stored_annotation = (
        explain_task.annotation if isinstance(explain_task, ExplainOutcome) else None
    )
    initial_drawing = (
        None
        if stored_annotation is None
        else stored_annotation.raw.model_dump(mode="json")
    )

    with left:
        annotated_canvas_image = load_study_image(annotated_image)

        original_canvas_width, original_canvas_height = annotated_canvas_image.size
        canvas_scale = 0.23  # laptop
        # canvas_scale = 0.42  # desktop
        canvas_width = round(original_canvas_width * canvas_scale)
        canvas_height = round(original_canvas_height * canvas_scale)
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
        st_canvas(
            stroke_width=8,
            stroke_color=DIFFERENCE_LABEL_STYLES[label]["color"],
            background_image=annotated_canvas_image,  # pyright: ignore[reportArgumentType]
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            display_toolbar=True,
            key=canvas_state,
            initial_drawing=initial_drawing,  # pyright: ignore[reportArgumentType]
        )
        patch_canvas_toolbar(canvas_state)
        current_canvas_state = st.session_state.get(canvas_state)

    explanation_key = (
        f"explain_text_{active.challenge.id}_{state.task_idx}_"
        f"{explain_task.annotated_image}"
    )
    explanation = st.text_input(
        "Optional Explanation",
        value=explain_task.explanation or ""
        if isinstance(explain_task, ExplainOutcome)
        else "",
        width="stretch",
        placeholder="Describe the visible difference from the references.",
        key=explanation_key,
        max_chars=244,
    )

    def save() -> None:
        annotation_payload = _build_annotation_payload(current_canvas_state)
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
            annotation=annotation_payload,
        )
        storage.upsert_explain_outcome(outcome)

        active.challenge.tasks[state.task_idx] = outcome
        state.active_challenge = active
        st.session_state.pop(canvas_state, None)
        st.session_state.pop(explanation_key, None)
        state.toaster = "Explanation saved."

    return save


def _render_rate_task(user: User, task: ActiveRateChallenge) -> Callable[[], None]:
    st.error("Rate task rendering not implemented yet.")
    return lambda: None


def _build_annotation_payload(
    canvas_state: Any | None,
) -> Annotation | None:
    try:
        annotation = Annotation.model_validate(canvas_state)
    except ValidationError:
        st.warning("Invalid canvas state, unable to save annotation.")
        return None

    if not annotation.has_objects:
        return None

    return annotation
