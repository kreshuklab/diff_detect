import base64
import os
import random
from html import escape
from io import BytesIO
from typing import Any, Callable

import numpy as np
import streamlit as st
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import ValidationError
from st_clickable_images import clickable_images
from streamlit_drawable_canvas import st_canvas

from diff_detect._patch_canvas_toolbar import patch_canvas_toolbar
from diff_detect.common import (
    CHALLENGE_NAMES,
    DIFFERENCE_LABEL_STYLES,
    DIFFERENCE_LABELS,
    EXPLAIN_CANVAS_SCALE,
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
from .models import (
    Image as StudyImage,
)


def _image_data_url(image: PILImage.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


def _scale_annotation(annotation: Annotation, scale: float) -> list[dict[str, Any]]:
    def scaled(value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value * scale
        return value

    objects = annotation.raw.model_dump(mode="json", exclude_unset=True).get(
        "objects", []
    )
    scaled_objects = []
    for obj in objects:
        scaled_obj = dict(obj)
        for key in ("left", "top", "width", "height", "radius", "rx", "ry"):
            if key in scaled_obj:
                scaled_obj[key] = scaled(scaled_obj[key])
        if isinstance(scaled_obj.get("path"), list):
            scaled_obj["path"] = [
                [part if idx == 0 else scaled(part) for idx, part in enumerate(segment)]
                for segment in scaled_obj["path"]
            ]
        if isinstance(scaled_obj.get("pathOffset"), dict):
            scaled_obj["pathOffset"] = {
                key: scaled(value) for key, value in scaled_obj["pathOffset"].items()
            }
        scaled_objects.append(scaled_obj)
    return scaled_objects


def _annotation_figure(image: PILImage.Image, annotation: Annotation | None):
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath
    from skimage.measure import label, regionprops

    def _draw_path_on_mask(
        draw: ImageDraw.ImageDraw,
        path_segments: list[Any],
        pixel_width: int,
    ) -> None:
        current_point: tuple[float, float] | None = None
        first_point: tuple[float, float] | None = None

        for segment in path_segments:
            if not segment:
                continue

            command = segment[0]
            coords = [float(value) for value in segment[1:]]
            next_point: tuple[float, float] | None = None

            if command == "M" and len(coords) >= 2:
                next_point = (coords[0], coords[1])
                first_point = next_point
            elif command == "L" and len(coords) >= 2:
                next_point = (coords[0], coords[1])
            elif command == "Q" and len(coords) >= 4:
                next_point = (coords[2], coords[3])
            elif command == "C" and len(coords) >= 6:
                next_point = (coords[4], coords[5])
            elif (
                command == "Z" and current_point is not None and first_point is not None
            ):
                draw.line([current_point, first_point], fill=255, width=pixel_width)
                current_point = first_point
                continue

            if next_point is None:
                continue
            if current_point is not None:
                draw.line([current_point, next_point], fill=255, width=pixel_width)
            current_point = next_point

    def _connected_path_rectangles(
        objects: list[dict[str, Any]],
        canvas_width: int,
        canvas_height: int,
    ) -> list[tuple[float, float, float, float, str, float]]:
        def _merge_close_boxes(
            boxes: list[tuple[float, float, float, float]],
            tolerance: float,
        ) -> list[tuple[float, float, float, float]]:
            if not boxes:
                return []

            remaining = [list(box) for box in boxes]
            merged: list[tuple[float, float, float, float]] = []

            while remaining:
                left, top, right, bottom = remaining.pop()
                changed = True
                while changed:
                    changed = False
                    next_remaining: list[list[float]] = []
                    for box_left, box_top, box_right, box_bottom in remaining:
                        overlaps_x = not (
                            box_left > right + tolerance or box_right < left - tolerance
                        )
                        overlaps_y = not (
                            box_top > bottom + tolerance or box_bottom < top - tolerance
                        )
                        if overlaps_x and overlaps_y:
                            left = min(left, box_left)
                            top = min(top, box_top)
                            right = max(right, box_right)
                            bottom = max(bottom, box_bottom)
                            changed = True
                        else:
                            next_remaining.append(
                                [box_left, box_top, box_right, box_bottom]
                            )
                    remaining = next_remaining

                merged.append((left, top, right, bottom))

            return merged

        masks_by_style: dict[tuple[str, float], PILImage.Image] = {}

        for obj in objects:
            if str(obj.get("type", "")).lower() != "path":
                continue

            path_segments = obj.get("path")
            if not isinstance(path_segments, list) or len(path_segments) == 0:
                continue

            stroke = str(obj.get("stroke") or "#e83e8c")
            line_width = max(1.5, min(float(obj.get("strokeWidth") or 3), 5))
            style_key = (stroke, line_width)
            if style_key not in masks_by_style:
                masks_by_style[style_key] = PILImage.new(
                    "L", (canvas_width, canvas_height), 0
                )

            draw = ImageDraw.Draw(masks_by_style[style_key])
            _draw_path_on_mask(draw, path_segments, max(1, round(line_width)))

        rectangles: list[tuple[float, float, float, float, str, float]] = []
        for (stroke, line_width), mask in masks_by_style.items():
            labeled = label((np.asarray(mask) > 0).astype(np.uint8), connectivity=2)
            component_boxes: list[tuple[float, float, float, float]] = []
            for region in regionprops(labeled):
                min_row, min_col, max_row, max_col = region.bbox
                left = float(min_col)
                top = float(min_row)
                right = float(max_col)
                bottom = float(max_row)
                if right <= left or bottom <= top:
                    continue
                component_boxes.append((left, top, right, bottom))

            merged_boxes = _merge_close_boxes(
                component_boxes,
                tolerance=max(1.0, line_width / 2),
            )
            for left, top, right, bottom in merged_boxes:
                rectangles.append(
                    (left, top, right - left, bottom - top, stroke, line_width)
                )

        return rectangles

    width, height = image.size
    fig = Figure(figsize=(4, 4 * height / width), dpi=150)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    if annotation is None:
        return fig

    scaled_objects = _scale_annotation(annotation, 1 / EXPLAIN_CANVAS_SCALE)
    convert_paths_to_rectangles = os.getenv("CONVERT_PATH_TO_RECT", "1") == "1"
    if convert_paths_to_rectangles:
        for (
            left,
            top,
            rect_width,
            rect_height,
            stroke,
            line_width,
        ) in _connected_path_rectangles(scaled_objects, width, height):
            ax.add_patch(
                Rectangle(
                    (left, top),
                    rect_width,
                    rect_height,
                    fill=False,
                    edgecolor=stroke,
                    linewidth=line_width,
                )
            )

    for obj in scaled_objects:
        stroke = obj.get("stroke") or "#e83e8c"
        line_width = max(1.5, min(float(obj.get("strokeWidth") or 3), 5))
        object_type = str(obj.get("type", "")).lower()
        if object_type == "rect":
            ax.add_patch(
                Rectangle(
                    (float(obj.get("left") or 0), float(obj.get("top") or 0)),
                    float(obj.get("width") or 0) * float(obj.get("scaleX") or 1),
                    float(obj.get("height") or 0) * float(obj.get("scaleY") or 1),
                    fill=False,
                    edgecolor=stroke,
                    linewidth=line_width,
                )
            )
        elif convert_paths_to_rectangles and object_type == "path":
            continue
        elif object_type == "path" and isinstance(obj.get("path"), list):
            vertices = []
            codes = []
            for segment in obj["path"]:
                if not segment:
                    continue
                command = segment[0]
                coords = [float(value) for value in segment[1:]]
                if command == "M" and len(coords) >= 2:
                    vertices.append((coords[0], coords[1]))
                    codes.append(MplPath.MOVETO)
                elif command == "L" and len(coords) >= 2:
                    vertices.append((coords[0], coords[1]))
                    codes.append(MplPath.LINETO)
                elif command == "Q" and len(coords) >= 4:
                    vertices.extend([(coords[0], coords[1]), (coords[2], coords[3])])
                    codes.extend([MplPath.CURVE3, MplPath.CURVE3])
                elif command == "C" and len(coords) >= 6:
                    vertices.extend(
                        [
                            (coords[0], coords[1]),
                            (coords[2], coords[3]),
                            (coords[4], coords[5]),
                        ]
                    )
                    codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                elif command == "Z":
                    vertices.append((0, 0))
                    codes.append(MplPath.CLOSEPOLY)
            if vertices:
                ax.add_patch(
                    PathPatch(
                        MplPath(vertices, codes),
                        fill=False,
                        edgecolor=stroke,
                        linewidth=line_width,
                        capstyle=obj.get("strokeLineCap") or "round",
                        joinstyle=obj.get("strokeLineJoin") or "round",
                    )
                )
    return fig


@st.cache_data(max_entries=100, show_spinner=False)
def _rate_annotation_figure(image: StudyImage, annotation_json: str | None):
    annotation = (
        None
        if annotation_json is None
        else Annotation.model_validate_json(annotation_json)
    )
    return _annotation_figure(load_study_image(image), annotation)


def _explanations_legend() -> str:
    items = "".join(
        (
            '<span style="display:inline-flex;align-items:center;gap:0.35rem;'
            'font-size:0.9rem;white-space:nowrap;">'
            f'<span style="width:0.75rem;height:0.75rem;border-radius:2px;'
            f'display:inline-block;background:{DIFFERENCE_LABEL_STYLES[label]["color"]};"></span>'
            f"{escape(label.title())}</span>"
        )
        for label in DIFFERENCE_LABELS
    )
    return (
        '<div style="display:flex;align-items:center;gap:1rem;'
        'flex-wrap:wrap;margin:0.5rem 0 0.75rem;">'
        '<h3 style="margin:0;">Rate Explanations</h3>'
        f"{items}</div>"
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

    candidate_image_ids = explain_task.image_ids
    candidate_images = {
        image_id: get_image(
            active.challenge_data.datasets,
            explain_task.dataset_id,
            image_id,
        )
        for image_id in candidate_image_ids
    }
    candidate_key = "_".join(explain_task.candidate_key)
    odd_image_key = (
        f"explain_odd_image_{active.challenge.id}_{state.task_idx}_{candidate_key}"
    )
    selected_odd_image = st.session_state.get(odd_image_key)
    if selected_odd_image not in candidate_image_ids:
        selected_odd_image = (
            explain_task.annotated_image
            if isinstance(explain_task, ExplainOutcome)
            else None
        )

    st.header(CHALLENGE_NAMES[active.challenge.id])

    if selected_odd_image is None:
        st.subheader("Choose the unique specimen")
        clicked_image_idx = clickable_images(
            paths=[
                _image_data_url(load_study_image(candidate_images[image_id]))
                for image_id in candidate_image_ids
            ],
            titles=[f"Image {idx + 1}" for idx in range(len(candidate_image_ids))],
            div_style={
                "display": "flex",
                "gap": "0.75rem",
                "align-items": "flex-start",
            },
            img_style={
                "width": "32%",
                "height": "auto",
                "object-fit": "contain",
                "cursor": "pointer",
                "border-radius": "4px",
            },
            key=f"{odd_image_key}_clickable",
        )
        if 0 <= clicked_image_idx < len(candidate_image_ids):
            selected_odd_image = candidate_image_ids[clicked_image_idx]

        if selected_odd_image is not None:
            st.session_state[odd_image_key] = selected_odd_image
            st.rerun()

        return lambda: None

    reference_image_ids = explain_task.references_for(selected_odd_image)
    annotated_image = candidate_images[selected_odd_image]
    reference_images = [candidate_images[image_id] for image_id in reference_image_ids]

    canvas_state_base = (
        f"explain_canvas_{active.challenge.id}_{state.task_idx}_"
        f"{candidate_key}_{selected_odd_image}_state"
    )
    canvas_reset_key = f"{canvas_state_base}_reset_generation"
    canvas_generation = st.session_state.get(canvas_reset_key, 0)
    canvas_state = (
        canvas_state_base
        if canvas_generation == 0
        else f"{canvas_state_base}_{canvas_generation}"
    )

    left, right = st.columns([2, 1])
    with left, st.container(width="content", horizontal=True):
        st.subheader("Annotate visual biological differences")
        label_radio_key = f"explain_label_{active.challenge.id}_{state.task_idx}"

        label = st.radio(
            "Active difference label",
            DIFFERENCE_LABELS,
            label_visibility="collapsed",
            format_func=format_label,
            horizontal=True,
            key=label_radio_key,
        )
        if st.button(
            "Reset",
            help="Reset labels and image selection",
            icon=":material/delete:",
            key=(
                f"reset_labels_{active.challenge.id}_{state.task_idx}_"
                f"{selected_odd_image}"
            ),
            width="content",
        ):
            if (
                isinstance(explain_task, ExplainOutcome)
                and explain_task.annotated_image == selected_odd_image
            ):
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
            st.session_state.pop(odd_image_key, None)
            st.rerun()

    with right:
        st.subheader("References")

    left, right = st.columns([2, 1])
    with right:
        for ref_img in reference_images:
            st.image(load_study_image(ref_img), width="stretch")

    stored_annotation = (
        explain_task.annotation
        if isinstance(explain_task, ExplainOutcome)
        and explain_task.annotated_image == selected_odd_image
        else None
    )
    initial_drawing = (
        None
        if stored_annotation is None
        else stored_annotation.raw.model_dump(mode="json", exclude_unset=True)
    )

    with left:
        annotated_canvas_image = load_study_image(annotated_image)

        original_canvas_width, original_canvas_height = annotated_canvas_image.size
        canvas_scale = EXPLAIN_CANVAS_SCALE
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
        f"{candidate_key}_{selected_odd_image}"
    )
    explanation = st.text_input(
        "Optional Explanation",
        value=explain_task.explanation or ""
        if isinstance(explain_task, ExplainOutcome)
        and explain_task.annotated_image == selected_odd_image
        else "",
        width="stretch",
        placeholder="Describe the visible difference from the references.",
        key=explanation_key,
        max_chars=244,
    )

    def save() -> None:
        st.session_state[odd_image_key] = selected_odd_image
        annotation_payload = _build_annotation_payload(current_canvas_state)
        cleaned_explanation = explanation.strip()
        if annotation_payload is None and not cleaned_explanation:
            return

        outcome = ExplainOutcome(
            dataset_id=explain_task.dataset_id,
            annotated_image=selected_odd_image,
            reference_image1=reference_image_ids[0],
            reference_image2=reference_image_ids[1],
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


def _render_rate_task(user: User, active: ActiveRateChallenge) -> Callable[[], None]:
    rate_task = active.challenge.tasks[state.task_idx]
    candidate_image_ids = rate_task.candidate_key
    candidate_images = {
        image_id: get_image(
            active.challenge_data.datasets,
            rate_task.dataset_id,
            image_id,
        )
        for image_id in candidate_image_ids
    }

    st.header(CHALLENGE_NAMES[active.challenge.id])
    # st.subheader("Original image set")
    for col, image_id in zip(st.columns(3), candidate_image_ids):
        with col:
            st.image(load_study_image(candidate_images[image_id]), width="stretch")

    users_by_role = {"own": rate_task.own, "peer": rate_task.peer, "ai": rate_task.ai}
    candidates = {}
    for role, user_id in users_by_role.items():
        outcome = active.challenge_data.reference_explain_outcomes.get(
            (rate_task.selection_key, user_id)
        )
        if outcome is None:
            st.error("Missing explanation candidates for this rating task.")
            return lambda: None
        candidates[role] = outcome

    order_key = (
        f"rate_order_{active.challenge.id}_{state.task_idx}_"
        f"{'_'.join(rate_task.task_key)}_{rate_task.selection_index}_"
        f"{rate_task.own}_{rate_task.peer}_{rate_task.ai}"
    )
    order = st.session_state.get(order_key)
    if not isinstance(order, list) or sorted(order) != ["ai", "own", "peer"]:
        order = ["own", "peer", "ai"]
        random.shuffle(order)
        st.session_state[order_key] = order

    labels_by_role = {role: chr(ord("A") + idx) for idx, role in enumerate(order)}
    roles_by_label = {label: role for role, label in labels_by_role.items()}
    labels = list(roles_by_label)
    most_convincing_key = f"rate_most_convincing_{active.challenge.id}_{state.task_idx}"
    most_likely_ai_key = f"rate_most_likely_ai_{active.challenge.id}_{state.task_idx}"

    def selected_user(key: str) -> str | None:
        label = st.session_state.get(key)
        if label is None:
            return None
        return users_by_role[roles_by_label[label]]

    def save_current_rating(*, toast: bool = False) -> None:
        most_convincing = selected_user(most_convincing_key)
        most_likely_ai = selected_user(most_likely_ai_key)
        if most_convincing is None and most_likely_ai is None:
            return

        outcome = RateOutcome(
            dataset_id=rate_task.dataset_id,
            annotated_image=rate_task.annotated_image,
            reference_image1=rate_task.reference_image1,
            reference_image2=rate_task.reference_image2,
            own=rate_task.own,
            peer=rate_task.peer,
            ai=rate_task.ai,
            most_convincing=most_convincing,
            most_likely_ai=most_likely_ai,
        )
        storage.upsert_rate_outcome(outcome)
        active.challenge.tasks[state.task_idx] = outcome
        state.active_challenge = active
        if toast:
            state.toaster = "Rating saved."

    st.markdown(_explanations_legend(), unsafe_allow_html=True)
    for col, role in zip(st.columns(3), order):
        candidate = candidates[role]
        annotated_image = get_image(
            active.challenge_data.datasets,
            candidate.dataset_id,
            candidate.annotated_image,
        )
        annotation_json = (
            None
            if candidate.annotation is None
            else candidate.annotation.model_dump_json(exclude_unset=True)
        )

        with col:
            st.markdown(f"#### Explanation {labels_by_role[role]}")
            st.pyplot(
                _rate_annotation_figure(annotated_image, annotation_json),
                clear_figure=False,
                width="stretch",
                pad_inches=0,
            )
            st.write(candidate.explanation or "")

    def selected_index(choice: str | None) -> int | None:
        if choice is None:
            return None
        for idx, label in enumerate(labels):
            if users_by_role[roles_by_label[label]] == choice:
                return idx
        return None

    existing_most_convincing = (
        rate_task.most_convincing if isinstance(rate_task, RateOutcome) else None
    )
    existing_most_likely_ai = (
        rate_task.most_likely_ai if isinstance(rate_task, RateOutcome) else None
    )
    st.divider()
    rating_container = st.container(
        width="stretch", horizontal=True, horizontal_alignment="center"
    )
    with rating_container:
        st.radio(
            "Which explanation is most convincing?",
            labels,
            index=selected_index(existing_most_convincing),
            horizontal=True,
            key=most_convincing_key,
            on_change=save_current_rating,
            width="content",
        )
        st.radio(
            "Which explanation was created by AI?",
            labels,
            index=selected_index(existing_most_likely_ai),
            horizontal=True,
            key=most_likely_ai_key,
            on_change=save_current_rating,
            width="content",
        )

    def save() -> None:
        save_current_rating(toast=True)

    return save


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
