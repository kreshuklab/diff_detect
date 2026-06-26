from study_storage import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    build_rating_options,
    canvas_has_objects,
    choose_rounds,
    load_rounds,
    load_seeded_annotations,
    load_wing_image,
)


def test_choose_rounds_is_stable_for_user():
    rounds = load_rounds()
    first = [task["task_id"] for task in choose_rounds(rounds, "ada")]
    second = [task["task_id"] for task in choose_rounds(rounds, "ada")]
    assert first == second
    assert len(first) == 3


def test_placeholder_images_match_canvas_size():
    image = load_wing_image({"image_id": "missing_fixture", "path": "data/images/missing_fixture.png"})
    assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)


def test_canvas_has_objects_requires_at_least_one_object():
    assert not canvas_has_objects(None)
    assert not canvas_has_objects({"objects": []})
    assert canvas_has_objects({"objects": [{"type": "path"}]})


def test_rating_options_include_self_peer_and_ai_with_fallback():
    task = load_rounds()[0]
    selected_id = task["odd_image_id"]
    own_submission = {
        "id": 1,
        "selected_image_id": selected_id,
        "label": "shape",
        "explanation": "",
        "composite_png_base64": "placeholder",
    }

    options = build_rating_options(task, own_submission, None, load_seeded_annotations(), "ada")

    assert sorted(option.source for option in options) == ["ai", "peer", "self"]


def test_round_manifest_uses_non_hybrid_mimic_groups_and_includes_strict_example():
    strict_rounds = 0
    for task in load_rounds():
        images = task["images"]
        references = [image for image in images if image["species_role"] == "reference"]
        odd = [image for image in images if image["species_role"] == "odd"]

        assert len(images) == 4
        assert len(references) == 3
        assert len(odd) == 1
        assert {image["hybrid_stat"] for image in images} == {"non-hybrid"}
        assert len({image["species"] for image in references}) == 1
        assert len({image["subspecies"] for image in references}) == 1
        assert len({image["view"] for image in images}) == 1
        assert len({image["mimic_group"] for image in images}) == 1
        assert odd[0]["species"] not in {image["species"] for image in references}
        if len({image["subspecies"] for image in images}) == 1:
            strict_rounds += 1

    assert strict_rounds >= 1
