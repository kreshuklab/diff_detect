from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DifferenceLabel = Literal["shape", "color", "texture"]
SpeciesRole = Literal["reference", "odd"]
WinnerSource = Literal["self", "peer", "ai"]
UserRole = Literal["participant", "maintainer"]


class RoundImageSource(BaseModel):
    camid: str = ""
    filename: str = ""
    filepath: str = ""
    md5: str = ""
    record_number: str = ""
    zenodo_link: str = ""


class RoundImage(BaseModel):
    image_id: str
    path: str
    source_url: str | None = None
    species_role: SpeciesRole = "reference"
    species: str = ""
    subspecies: str = ""
    view: str = ""
    mimic_group: str = ""
    hybrid_stat: str = ""
    source: RoundImageSource = Field(default_factory=RoundImageSource)


class RoundMetadata(BaseModel):
    dataset: str = ""
    dataset_url: str = ""
    manifest_url: str = ""
    view: str = ""
    generation_seed: int | None = None
    generation_rule: str = ""
    round_rule: str = ""
    reference_species: str = ""
    reference_subspecies: str = ""
    odd_species: str = ""
    odd_subspecies: str = ""
    mimic_group: str = ""
    hybrid_stat: str = ""
    references_share_subspecies: bool | None = None
    all_four_share_subspecies: bool | None = None
    all_four_share_mimic_group: bool | None = None


class Round(BaseModel):
    task_id: str
    odd_image_id: str
    images: list[RoundImage]
    metadata: RoundMetadata = Field(default_factory=RoundMetadata)

    @model_validator(mode="after")
    def validate_round_images(self) -> "Round":
        odd_images = [image for image in self.images if image.species_role == "odd"]
        if len(self.images) != 4:
            raise ValueError("A round must contain exactly four images.")
        if len(odd_images) != 1:
            raise ValueError("A round must contain exactly one odd image.")
        if odd_images[0].image_id != self.odd_image_id:
            raise ValueError("odd_image_id must match the image marked as odd.")
        return self


class SeededAnnotation(BaseModel):
    task_id: str
    source: Literal["ai", "peer"]
    selected_image_id: str
    label: DifferenceLabel
    explanation: str = ""
    annotation_color: str = "#ffb000"


class AnnotationLayers(BaseModel):
    mode: Literal["single_canvas_color_coded_labels"]
    labels: list[DifferenceLabel]
    canvas_json: dict[str, Any]


class SubmissionPayload(BaseModel):
    username: str
    task_id: str
    selected_image_id: str
    label: DifferenceLabel
    labels: list[DifferenceLabel]
    explanation: str | None = None
    canvas_json: dict[str, Any]
    annotation_layers: AnnotationLayers
    composite_png_base64: str

    @field_validator("labels")
    @classmethod
    def require_labels(cls, labels: list[DifferenceLabel]) -> list[DifferenceLabel]:
        if not labels:
            raise ValueError("At least one difference label is required.")
        return labels


class RatingOption(BaseModel):
    option_id: str
    source: WinnerSource
    task_id: str
    selected_image_id: str
    label: str
    explanation: str
    composite_png_base64: str
    submission_id: str | None = None


class RatingPayload(BaseModel):
    username: str
    task_id: str
    winner_source: WinnerSource
    winner_submission_id: str | None = None
    option_payload: list[dict[str, Any]]
