from models import CanvasJson, Round
from study_storage import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DATASET_ID,
    DIFFERENCE_LABEL_STYLES,
    N_ROUNDS,
    build_rating_options,
    canvas_has_objects,
    canvas_labels,
    choose_rounds,
    load_image,
    load_rounds,
    load_seeded_annotations,
)


def test_choose_rounds_is_stable_for_user():
    rounds = load_rounds()
    first = [task.task_id for task in choose_rounds(rounds, "ada")]
    second = [task.task_id for task in choose_rounds(rounds, "ada")]
    assert first == second
    assert len(first) == N_ROUNDS


def test_default_rounds_use_active_dataset_id():
    assert {task.metadata.dataset_id for task in load_rounds()} == {DATASET_ID}


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
    assert {option.dataset_id for option in options} == {DATASET_ID}


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
