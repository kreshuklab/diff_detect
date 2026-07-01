from sqlmodel import SQLModel, create_engine

from diff_detect.models import (
    DatasetId,
    ExplainChallenge,
    ExplainOutcome,
    ExplainTask,
    RateChallenge,
    RateOutcome,
    RateTask,
    User,
    UserKind,
    UserRole,
)
from diff_detect.pages._builder import _build_annotation_payload
from diff_detect.storage_sqlite import SqliteStorage


def test_build_annotation_payload_preserves_canvas_json_and_labels():
    canvas_json = {
        "version": "4.4.0",
        "objects": [{"type": "path", "stroke": "#e83e8c"}],
    }

    payload = _build_annotation_payload(canvas_json, "shape")

    assert payload == {
        "mode": "single_canvas_color_coded_labels",
        "labels": ["color"],
        "canvas_json": canvas_json,
    }


def test_build_annotation_payload_uses_fallback_label_for_uncolored_strokes():
    payload = _build_annotation_payload(
        {"objects": [{"type": "path"}]},
        "texture",
    )

    assert payload is not None
    assert payload["labels"] == ["texture"]


def test_build_annotation_payload_returns_none_without_objects():
    assert _build_annotation_payload({"objects": []}, "shape") is None
    assert _build_annotation_payload(None, "shape") is None


def test_sqlite_storage_upserts_explain_outcome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    storage.add_user(
        User(
            id="ada",
            kind=UserKind.HUMAN,
            role=UserRole.PARTICIPANT,
            hashed_password="hash",
        )
    )
    first = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
        user="ada",
        explanation="first",
        annotations=None,
    )
    second = first.model_copy(update={"explanation": "second"})

    storage.upsert_explain_outcome(first)
    storage.upsert_explain_outcome(second)

    outcomes = storage.fetch_explain_outcomes("ada")
    assert len(outcomes) == 1
    assert outcomes[0].explanation == "second"


def test_challenge_progress_counts_matching_outcome_type():
    explain_challenge = ExplainChallenge(
        id="explain_dummy",
        tasks=[
            ExplainOutcome(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/a",
                reference_image1="butterfly/b",
                reference_image2="butterfly/c",
                user="ada",
                explanation="done",
                annotations=None,
            ),
            ExplainTask(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/d",
                reference_image1="butterfly/e",
                reference_image2="butterfly/f",
            ),
        ],
    )
    rate_challenge = RateChallenge(
        id="rate_dummy",
        tasks=[
            RateOutcome(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/a",
                reference_image1="butterfly/b",
                reference_image2="butterfly/c",
                own="ada",
                peer="grace",
                ai="bot",
                most_convincing="ada",
                most_likely_ai="bot",
            ),
            RateTask(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/d",
                reference_image1="butterfly/e",
                reference_image2="butterfly/f",
                own="ada",
                peer="grace",
                ai="bot",
            ),
        ],
    )

    assert explain_challenge.done_count == 1
    assert explain_challenge.first_undone == 1
    assert rate_challenge.done_count == 1
    assert rate_challenge.first_undone == 1
