from diff_detect.models import ImageKey, SelectionChoice
from diff_detect.models_old import CanvasJson
from diff_detect.storage_supabase import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DATASET_ID_ENV_VAR,
    DEFAULT_CHALLENGE_ID,
    DEFAULT_DATASET_ID,
    DIFFERENCE_LABEL_STYLES,
    available_dataset_ids,
    build_rating_options,
    canvas_has_objects,
    canvas_labels,
    choose_rounds,
    completed_task_ids,
    configured_dataset_id,
    load_dataset,
    load_image,
    load_rounds,
    load_seeded_annotations,
    load_selection_challenge,
    normalize_dataset_id,
)


def test_choose_rounds_is_stable_for_user():
    rounds = load_rounds()
    first = [task.task_id for task in choose_rounds(rounds, "ada", DEFAULT_DATASET_ID)]
    second = [task.task_id for task in choose_rounds(rounds, "ada", DEFAULT_DATASET_ID)]
    assert first == second
    assert set(first) == {task.task_id for task in rounds}
    assert len(first) == len(rounds)


def test_new_selection_choice_model_accepts_manifest_image_ids():
    task = load_rounds()[0]
    images = tuple(
        ImageKey(dataset_id=DEFAULT_DATASET_ID, image_id=image.image_id)
        for image in task.images
    )

    choice = SelectionChoice(
        images=images,
        user="ada",
        index=0,
        user_kind="human",
        annotations=[{"labels": ["shape"]}],
    )

    assert choice.images == images


def test_completed_task_ids_ignore_duplicates_and_unknown_tasks():
    rows = [
        {"task_id": "round_1"},
        {"task_id": "round_1"},
        {"task_id": "old_round"},
        {"task_id": None},
        {},
    ]

    assert completed_task_ids(rows, {"round_1", "round_2"}) == {"round_1"}


def test_default_rounds_use_active_dataset_id():
    assert {task.metadata.dataset_id for task in load_rounds()} == {DEFAULT_DATASET_ID}


def test_configured_dataset_id_uses_environment_override(monkeypatch):
    monkeypatch.setenv(DATASET_ID_ENV_VAR, "env_dataset")
    assert configured_dataset_id("explicit_dataset") == "env_dataset"
    assert configured_dataset_id() == "env_dataset"
    monkeypatch.delenv(DATASET_ID_ENV_VAR)
    assert configured_dataset_id("explicit_dataset") == "explicit_dataset"


def test_normalize_dataset_id_rejects_paths():
    assert normalize_dataset_id(" butterfly ") == DEFAULT_DATASET_ID
    for dataset_id in ("", "../other", "nested/dataset", "."):
        try:
            normalize_dataset_id(dataset_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {dataset_id!r} to be rejected")


def test_available_dataset_ids_include_default_dataset():
    assert DEFAULT_DATASET_ID in available_dataset_ids()


def test_default_dataset_and_challenge_use_new_models():
    dataset = load_dataset()
    challenge = load_selection_challenge()

    assert dataset.root
    assert challenge.dataset_id == DEFAULT_DATASET_ID
    assert challenge.challenge_id == DEFAULT_CHALLENGE_ID
    assert challenge.tasks


def test_seeded_annotations_are_optional_for_datasets(tmp_path):
    assert load_seeded_annotations(path=tmp_path / "seeded_annotations.json") == []


def test_placeholder_images_match_canvas_size():
    image = load_image(
        {"image_id": "missing_fixture", "path": "data/test/missing_fixture.png"}
    )
    assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)


def test_canvas_has_objects_requires_at_least_one_object():
    assert not canvas_has_objects(None)
    assert not canvas_has_objects({"objects": []})
    assert canvas_has_objects({"objects": [{"type": "path"}]})


def test_canvas_labels_are_derived_from_stroke_colors():
    canvas_json = {
        "objects": [
            {"type": "path", "stroke": DIFFERENCE_LABEL_STYLES["shape"]["color"]},
            {"type": "path", "stroke": "#e83e8c"},
            {"type": "path", "stroke": DIFFERENCE_LABEL_STYLES["shape"]["color"]},
        ]
    }

    assert canvas_labels(canvas_json) == ["shape", "color"]


def test_canvas_labels_returns_multiple_stroke_labels():
    labels = canvas_labels(
        {
            "objects": [
                {"type": "path", "stroke": DIFFERENCE_LABEL_STYLES["shape"]["color"]},
                {"type": "path", "stroke": "#e83e8c"},
                {"type": "path", "stroke": "#006d77"},
            ]
        }
    )
    assert labels == ["shape", "color", "texture"]


def test_canvas_json_model_preserves_fabric_object_extras():
    canvas_json = CanvasJson.model_validate(
        {
            "version": "4.4.0",
            "objects": [
                {
                    "type": "path",
                    "stroke": DIFFERENCE_LABEL_STYLES["shape"]["color"],
                    "path": [["M", 1, 2], ["L", 3, 4]],
                }
            ],
        }
    )

    assert canvas_has_objects(canvas_json)
    assert canvas_labels(canvas_json) == ["shape"]
    assert canvas_json.objects[0].model_extra == {"path": [["M", 1, 2], ["L", 3, 4]]}


def test_rating_options_include_self_without_seeded_fallbacks():
    task = load_rounds()[0]
    selected_id = task.odd_image_id
    own_submission = {
        "id": 1,
        "selected_image_id": selected_id,
        "labels": ["shape"],
        "explanation": "",
        "composite_png_base64": "placeholder",
    }

    options = build_rating_options(
        task, own_submission, None, load_seeded_annotations(), "ada"
    )

    assert sorted(option.source for option in options) == ["self"]
    assert {option.dataset_id for option in options} == {DEFAULT_DATASET_ID}


def test_round_views_are_built_from_selection_challenge_tasks():
    for task in load_rounds():
        images = task.images

        assert len(images) >= 1
        assert task.task_id.startswith(f"{DEFAULT_CHALLENGE_ID}:")
        assert task.metadata.dataset_id == DEFAULT_DATASET_ID
