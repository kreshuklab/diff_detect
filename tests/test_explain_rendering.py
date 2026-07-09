from collections import Counter

from sqlalchemy import text
from sqlmodel import SQLModel, create_engine

from diff_detect._leaderboard_page import _score_explain, _score_rate
from diff_detect._state import state
from diff_detect._storage._storage_sqlite import SqliteStorage
from diff_detect._task_page import _build_annotation_payload, _scale_annotation
from diff_detect.ai_annotations import (
    import_ai_annotations,
    iter_ai_annotation_outcomes,
)
from diff_detect.challenges import get_explain_challenge
from diff_detect.models import (
    ActiveExplainChallenge,
    Annotation,
    ChallengeData,
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


def _user(user_id: str, *, kind: UserKind = UserKind.HUMAN) -> User:
    return User(
        id=user_id,
        name=user_id.title(),
        lab="lab",
        kind=kind,
        role=UserRole.PARTICIPANT,
        hashed_password="hash",
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


def test_scale_annotation_scales_geometry_only():
    annotation = Annotation.model_validate(
        {
            "raw": {
                "version": "4.4.0",
                "objects": [
                    {
                        "type": "rect",
                        "left": 1,
                        "top": 2,
                        "width": 3,
                        "height": 4,
                        "strokeWidth": 8,
                    },
                    {
                        "type": "path",
                        "path": [["M", 1, 2], ["Q", 3, 4, 5, 6]],
                    },
                ],
            }
        }
    )

    rect, path = _scale_annotation(annotation, 10)

    assert rect["left"] == 10
    assert rect["top"] == 20
    assert rect["width"] == 30
    assert rect["height"] == 40
    assert rect["strokeWidth"] == 8
    assert path["path"] == [["M", 10, 20], ["Q", 30, 40, 50, 60]]


def test_explain_challenge_loads_csv_without_ai_annotations():
    datasets, challenge = get_explain_challenge("explain_butterfly_easy")

    assert challenge.task_count == 10
    assert len(datasets[DatasetId.BUTTERFLY].images) == 30
    first_task = challenge.tasks[0]
    assert isinstance(first_task, ExplainTask)
    assert first_task.annotated_image.startswith("triple_")
    first_image = datasets[DatasetId.BUTTERFLY].images[first_task.annotated_image]
    assert first_image.source.startswith("butterfly/download/triple_")
    assert first_image.image_info["camid"].startswith("CAM")


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


def test_sqlite_storage_upserts_rate_outcome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    for user in (_user("ada"), _user("grace"), _user("bot", kind=UserKind.AI)):
        storage.add_user(user)

    first = RateOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
        own="ada",
        peer="grace",
        ai="bot",
        most_convincing="ada",
        most_likely_ai="bot",
    )
    second = first.model_copy(update={"most_convincing": "grace"})

    storage.upsert_rate_outcome(first)
    storage.upsert_rate_outcome(second)

    outcomes = storage.fetch_rate_outcomes("ada")
    assert len(outcomes) == 1
    assert outcomes[0].most_convincing == "grace"
    assert outcomes[0].most_likely_ai == "bot"


def test_sqlite_storage_saves_partial_rate_outcome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    for user in (_user("ada"), _user("grace"), _user("bot", kind=UserKind.AI)):
        storage.add_user(user)

    storage.upsert_rate_outcome(
        RateOutcome(
            dataset_id=DatasetId.BUTTERFLY,
            annotated_image="butterfly/a",
            reference_image1="butterfly/b",
            reference_image2="butterfly/c",
            own="ada",
            peer="grace",
            ai="bot",
            most_convincing="grace",
            most_likely_ai=None,
        )
    )

    outcome = storage.fetch_rate_outcomes("ada")[0]
    assert outcome.most_convincing == "grace"
    assert outcome.most_likely_ai is None
    assert not outcome.complete


def test_sqlite_storage_migrates_rate_choices_to_nullable(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE rateoutcome (
                    dataset_id VARCHAR(9) NOT NULL,
                    annotated_image VARCHAR NOT NULL,
                    reference_image1 VARCHAR NOT NULL,
                    reference_image2 VARCHAR NOT NULL,
                    own VARCHAR NOT NULL,
                    peer VARCHAR NOT NULL,
                    ai VARCHAR NOT NULL,
                    timestamp DATETIME NOT NULL,
                    most_convincing VARCHAR NOT NULL,
                    most_likely_ai VARCHAR NOT NULL,
                    PRIMARY KEY (
                        annotated_image, reference_image1, reference_image2,
                        own, peer, ai
                    )
                )
                """
            )
        )

    storage = SqliteStorage(engine=engine)
    storage.upsert_rate_outcome(
        RateOutcome(
            dataset_id=DatasetId.BUTTERFLY,
            annotated_image="butterfly/a",
            reference_image1="butterfly/b",
            reference_image2="butterfly/c",
            own="ada",
            peer="grace",
            ai="bot",
            most_convincing=None,
            most_likely_ai="bot",
        )
    )

    outcome = storage.fetch_rate_outcomes("ada")[0]
    assert outcome.most_convincing is None
    assert outcome.most_likely_ai == "bot"


def test_explain_task_candidate_key_ignores_chosen_odd_image():
    base_task = ExplainTask(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
    )
    outcome = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/b",
        reference_image1="butterfly/a",
        reference_image2="butterfly/c",
        user="ada",
        explanation="odd one changed",
        annotation=None,
    )

    assert base_task.task_key != outcome.task_key
    assert base_task.candidate_key == outcome.candidate_key
    assert base_task.references_for("butterfly/b") == ("butterfly/a", "butterfly/c")


def test_selection_key_ignores_reference_order():
    first = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
        user="ada",
        explanation="first",
        annotation=None,
    )
    second = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/c",
        reference_image2="butterfly/b",
        user="ada",
        explanation="second",
        annotation=None,
    )

    assert first.selection_key == second.selection_key


def test_sqlite_storage_replaces_explain_outcome_when_odd_choice_changes(tmp_path):
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
        explanation="first odd choice",
        annotation=None,
    )
    second = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/b",
        reference_image1="butterfly/a",
        reference_image2="butterfly/c",
        user="ada",
        explanation="changed odd choice",
        annotation=None,
    )

    storage.upsert_explain_outcome(first)
    storage.upsert_explain_outcome(second)

    outcomes = storage.fetch_explain_outcomes("ada")
    assert len(outcomes) == 1
    assert outcomes[0].annotated_image == "butterfly/b"
    assert outcomes[0].explanation == "changed odd choice"


def test_reference_explain_outcome_matches_selection_key(
    tmp_path,
):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    storage.add_user(_user("ada"))
    storage.add_user(_user("grace"))
    storage.add_user(_user("marie"))
    own = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
        user="ada",
        explanation="own",
        annotation=None,
    )
    peer = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/b",
        reference_image1="butterfly/a",
        reference_image2="butterfly/c",
        user="grace",
        explanation="peer picked another odd image",
        annotation=None,
    )
    matching_peer = ExplainOutcome(
        dataset_id=DatasetId.BUTTERFLY,
        annotated_image="butterfly/a",
        reference_image1="butterfly/b",
        reference_image2="butterfly/c",
        user="marie",
        explanation="peer picked the same odd image",
        annotation=None,
    )

    storage.upsert_explain_outcome(own)
    storage.upsert_explain_outcome(peer)
    storage.upsert_explain_outcome(matching_peer)

    outcome = storage.fetch_random_reference_explain_outcome(own, UserKind.HUMAN)
    assert outcome is not None
    assert outcome.user == "marie"
    assert outcome.task_key == own.task_key
    assert outcome.candidate_key == own.candidate_key


def test_fetch_challenges_ignores_stale_active_challenge_when_enabling_rate(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    user = User(
        id="ada",
        name="Ada",
        lab="lab",
        kind=UserKind.HUMAN,
        role=UserRole.MAINTAINER,
        hashed_password="hash",
    )
    storage.add_user(user)

    datasets, stale_challenge = get_explain_challenge("explain_butterfly_easy")
    state.active_challenge = ActiveExplainChallenge(
        challenge_data=ChallengeData(
            datasets=datasets,
            reference_explain_outcomes={},
            challenges={stale_challenge.id: stale_challenge},
        ),
        challenge_id=stale_challenge.id,
    )
    try:
        for task in stale_challenge.tasks:
            storage.upsert_explain_outcome(
                ExplainOutcome(
                    dataset_id=task.dataset_id,
                    annotated_image=task.annotated_image,
                    reference_image1=task.reference_image1,
                    reference_image2=task.reference_image2,
                    user=user.id,
                    explanation="done",
                    annotation=None,
                )
            )

        fresh = storage.fetch_challenges(user)

        assert fresh.explain_challenges["explain_butterfly_easy"].finished
        assert fresh.rate_challenges["rate_butterfly_easy"].task_count == len(
            stale_challenge.tasks
        )
    finally:
        state.active_challenge = None


def test_fetch_challenges_keys_rate_outcomes_by_selection_key(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)
    user = User(
        id="ada",
        name="Ada",
        lab="lab",
        kind=UserKind.HUMAN,
        role=UserRole.MAINTAINER,
        hashed_password="hash",
    )
    storage.add_user(user)

    _, challenge = get_explain_challenge("explain_butterfly_easy")
    first_task = challenge.tasks[0]
    changed_selection = first_task.reference_image1
    changed_references = first_task.references_for(changed_selection)
    changed_outcome = ExplainOutcome(
        dataset_id=first_task.dataset_id,
        annotated_image=changed_selection,
        reference_image1=changed_references[0],
        reference_image2=changed_references[1],
        user=user.id,
        explanation="changed selection",
        annotation=None,
    )

    storage.upsert_explain_outcome(changed_outcome)
    for task in challenge.tasks[1:]:
        storage.upsert_explain_outcome(
            ExplainOutcome(
                dataset_id=task.dataset_id,
                annotated_image=task.annotated_image,
                reference_image1=task.reference_image1,
                reference_image2=task.reference_image2,
                user=user.id,
                explanation="done",
                annotation=None,
            )
        )
    storage.upsert_rate_outcome(
        RateOutcome(
            dataset_id=first_task.dataset_id,
            annotated_image=first_task.annotated_image,
            reference_image1=first_task.reference_image1,
            reference_image2=first_task.reference_image2,
            own=user.id,
            peer=user.id,
            ai=user.id,
            most_convincing=user.id,
            most_likely_ai=user.id,
        )
    )

    fresh = storage.fetch_challenges(user)
    first_rate_task = fresh.rate_challenges["rate_butterfly_easy"].tasks[0]

    assert not isinstance(first_rate_task, RateOutcome)
    assert first_rate_task.selection_key == changed_outcome.selection_key


def test_ai_annotation_parser_builds_bounding_box_outcomes():
    outcomes = list(iter_ai_annotation_outcomes())

    candidate_counts = Counter(outcome.candidate_key for outcome in outcomes)
    assert len(outcomes) == 60
    assert set(candidate_counts.values()) == {3}
    assert {outcome.user for outcome in outcomes} == {"ai_a", "ai_b", "ai_c"}
    first = outcomes[0]
    assert first.user == "ai_a"
    assert first.annotation is not None
    first_object = first.annotation.raw.objects[0]
    assert first_object.type == "rect"
    assert first_object.stroke == "#e83e8c"
    assert first_object.model_extra["ai_feature"]


def test_import_ai_annotations_saves_ai_user_and_outcomes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    SQLModel.metadata.create_all(engine)
    storage = SqliteStorage(engine=engine)

    outcomes = import_ai_annotations(storage)

    assert len(outcomes) == 60
    for user_id in ("ai_a", "ai_b", "ai_c"):
        ai_user = storage.fetch_user(user_id)
        assert ai_user is not None
        assert ai_user.kind == UserKind.AI
        assert len(storage.fetch_explain_outcomes(user_id)) == 20


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


def test_partial_rate_outcome_does_not_count_as_done():
    challenge = RateChallenge(
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
                most_likely_ai=None,
            ),
        ],
    )

    assert challenge.done_count == 0
    assert not challenge.finished
    assert challenge.first_undone == 0


def test_leaderboard_scores_explain_and_rate_by_user_and_lab():
    users = {
        "ada": _user("ada"),
        "grace": _user("grace"),
        "bot": _user("bot", kind=UserKind.AI),
    }
    challenge = ExplainChallenge(
        id="explain_dummy",
        tasks=[
            ExplainTask(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/a",
                reference_image1="butterfly/b",
                reference_image2="butterfly/c",
            ),
            ExplainTask(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/d",
                reference_image1="butterfly/e",
                reference_image2="butterfly/f",
            ),
        ],
    )
    explain_user_rows, explain_lab_rows = _score_explain(
        challenge,
        [
            ExplainOutcome(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/a",
                reference_image1="butterfly/b",
                reference_image2="butterfly/c",
                user="ada",
                explanation="correct",
                annotation=None,
            ),
            ExplainOutcome(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/f",
                reference_image1="butterfly/d",
                reference_image2="butterfly/e",
                user="grace",
                explanation="wrong",
                annotation=None,
            ),
            ExplainOutcome(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/a",
                reference_image1="butterfly/b",
                reference_image2="butterfly/c",
                user="bot",
                explanation="not a participant",
                annotation=None,
            ),
        ],
        users,
    )
    rate_user_rows, rate_lab_rows = _score_rate(
        challenge,
        [
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
            RateOutcome(
                dataset_id=DatasetId.BUTTERFLY,
                annotated_image="butterfly/d",
                reference_image1="butterfly/e",
                reference_image2="butterfly/f",
                own="grace",
                peer="ada",
                ai="bot",
                most_convincing="ada",
                most_likely_ai="ada",
            ),
        ],
        users,
    )

    assert explain_user_rows == [
        {
            "User": "Ada (ada)",
            "Score x/1": 1,
        },
        {
            "User": "Grace (grace)",
            "Score x/1": 0,
        },
    ]
    assert explain_lab_rows == [
        {
            "Lab": "lab",
            "Score x/2": 1,
        }
    ]
    assert rate_user_rows[0]["User"] == "Ada (ada)"
    assert rate_user_rows[0]["Score x/1"] == 1
    assert rate_lab_rows == [
        {
            "Lab": "lab",
            "Score x/2": 1,
        }
    ]
