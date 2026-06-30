from diff_detect.models_old import CanvasJson, Round
from diff_detect.study_storage import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DATASET_ID_ENV_VAR,
    DEFAULT_DATASET_ID,
    DIFFERENCE_LABEL_STYLES,
    available_dataset_ids,
    build_rating_options,
    canvas_has_objects,
    canvas_labels,
    choose_rounds,
    completed_task_ids,
    configured_dataset_id,
    load_image,
    load_rounds,
    load_seeded_annotations,
    normalize_dataset_id,
)


def test_choose_rounds_is_stable_for_user():
    rounds = load_rounds()
    first = [task.task_id for task in choose_rounds(rounds, "ada", DEFAULT_DATASET_ID)]
    second = [task.task_id for task in choose_rounds(rounds, "ada", DEFAULT_DATASET_ID)]
    assert first == second
    assert set(first) == {task.task_id for task in rounds}
    assert len(first) == len(rounds)


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
    assert normalize_dataset_id(" hf_heliconius ") == DEFAULT_DATASET_ID
    for dataset_id in ("", "../other", "nested/dataset", "."):
        try:
            normalize_dataset_id(dataset_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {dataset_id!r} to be rejected")


def test_available_dataset_ids_include_default_dataset():
    assert DEFAULT_DATASET_ID in available_dataset_ids()


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


def test_rating_options_include_self_peer_and_ai_with_fallback():
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

    assert sorted(option.source for option in options) == ["ai", "peer", "self"]
    assert {option.dataset_id for option in options} == {DEFAULT_DATASET_ID}


def test_round_model_allows_three_or_four_images():
    round_data = {
        "task_id": "three_image_fixture",
        "odd_image_id": "odd",
        "images": [
            {"image_id": "ref_1", "path": "data/test/ref_1.png"},
            {"image_id": "ref_2", "path": "data/test/ref_2.png"},
            {
                "image_id": "odd",
                "path": "data/test/odd.png",
                "species_role": "odd",
            },
        ],
    }

    assert len(Round.model_validate(round_data).images) == 3


def test_round_manifest_uses_non_hybrid_mimic_groups_and_includes_strict_example():
    strict_rounds = 0
    for task in load_rounds():
        images = task.images
        references = [image for image in images if image.species_role == "reference"]
        odd = [image for image in images if image.species_role == "odd"]

        assert len(images) in (3, 4)
        assert len(references) == len(images) - 1
        assert len(odd) == 1
        assert {image.hybrid_stat for image in images} == {"non-hybrid"}
        assert len({image.species for image in references}) == 1
        assert len({image.subspecies for image in references}) == 1
        assert len({image.view for image in images}) == 1
        assert len({image.mimic_group for image in images}) == 1
        assert odd[0].species not in {image.species for image in references}
        if len({image.subspecies for image in images}) == 1:
            strict_rounds += 1

    assert strict_rounds >= 1
