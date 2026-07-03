from sqlmodel import SQLModel, create_engine

from diff_detect._storage._storage_sqlite import SqliteStorage
from diff_detect._task_page import _build_annotation_payload
from diff_detect.models import (
    Annotation,
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


def test_build_annotation_payload_strips_canvas_image_data():
    raw = {
        "version": "4.4.0",
        "objects": [{"type": "path", "stroke": "#e83e8c"}],
    }

    payload = _build_annotation_payload(
        {"data": "data:image/png;base64,large-rendered-payload", "raw": raw}
    )

    assert isinstance(payload, Annotation)
    dumped = payload.model_dump(mode="json", exclude_none=True)
    assert dumped == {"raw": raw}
    assert "data" not in dumped


def test_build_annotation_payload_preserves_fabric_object_extras():
    raw = {
        "version": "4.4.0",
        "objects": [
            {
                "type": "path",
                "stroke": "#e83e8c",
                "path": [["M", 1, 2]],
            }
        ],
    }

    payload = _build_annotation_payload({"data": "image", "raw": raw})

    assert payload is not None
    assert payload.raw.objects[0].model_extra == {"path": [["M", 1, 2]]}


def test_build_annotation_payload_preserves_free_draw_path_fill():
    raw = {
        "version": "4.4.0",
        "objects": [
            {
                "type": "Path",
                "stroke": "#e83e8c",
                "fill": None,
            }
        ],
    }

    payload = _build_annotation_payload({"data": "image", "raw": raw})

    assert payload is not None
    dumped = payload.model_dump(mode="json", exclude_unset=True)
    assert dumped["raw"]["objects"] == [
        {"type": "Path", "stroke": "#e83e8c", "fill": None}
    ]


def test_build_annotation_payload_returns_none_without_objects():
    assert _build_annotation_payload({"data": "image", "raw": {"objects": []}}) is None
    assert _build_annotation_payload({"objects": [{"type": "path"}]}) is None
    assert _build_annotation_payload(None) is None


def test_sqlite_storage_upserts_explain_outcome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    storage.add_user(
        User(
            id="ada",
            name="Ada",
            lab="lab",
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
        annotation=None,
    )
    second = first.model_copy(update={"explanation": "second"})

    storage.upsert_explain_outcome(first)
    storage.upsert_explain_outcome(second)

    outcomes = storage.fetch_explain_outcomes("ada")
    assert len(outcomes) == 1
    assert outcomes[0].explanation == "second"


def test_sqlite_storage_round_trips_annotation_model(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    storage.add_user(
        User(
            id="ada",
            name="Ada",
            lab="lab",
            kind=UserKind.HUMAN,
            role=UserRole.PARTICIPANT,
            hashed_password="hash",
        )
    )
    raw = {
        "version": "4.4.0",
        "objects": [
            {
                "type": "path",
                "stroke": "#e83e8c",
                "fill": None,
                "path": [["M", 1, 2]],
            }
        ],
    }
    outcome = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
        user="ada",
        explanation=None,
        annotation=Annotation.model_validate({"data": "image", "raw": raw}),
    )

    storage.upsert_explain_outcome(outcome)

    stored_annotation = storage.fetch_explain_outcomes("ada")[0].annotation
    assert isinstance(stored_annotation, Annotation)


def test_sqlite_storage_deletes_explain_outcome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    storage.add_user(
        User(
            id="ada",
            name="Ada",
            lab="lab",
            kind=UserKind.HUMAN,
            role=UserRole.PARTICIPANT,
            hashed_password="hash",
        )
    )
    raw = {"version": "4.4.0", "objects": [{"type": "path", "stroke": "#e83e8c"}]}
    outcome = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
        user="ada",
        explanation=None,
        annotation=Annotation.model_validate({"data": "image", "raw": raw}),
    )

    storage.upsert_explain_outcome(outcome)
    assert storage.fetch_explain_outcomes("ada") == [outcome]
    storage.delete_explain_outcome(outcome)
    assert storage.fetch_explain_outcomes("ada") == []


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
                annotation=None,
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
