from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Annotated, Any, Literal, Sequence, Sized, TypeVar

from annotated_types import Ge, Le, MinLen, Predicate
from pydantic import BaseModel, RootModel, model_validator
from typing_extensions import Self

S = TypeVar("S", bound=Sized)
NonEmpty = Annotated[S, MinLen(1)]

_Id = Annotated[
    str,
    Predicate(lambda s: bool(s) and all(c.isalnum() or c in "_-" for c in s)),
]

UserId = str
DatasetId = _Id
ImageId = _Id
TaskId = _Id
ChallengeId = Literal["dummy", "butterfly_easy", "butterfly_difficult"]
# SelectionChoiceId = tuple[Unpack[SelectionTaskId], UserId, int]
# RatingChoiceId = tuple[Unpack[SelectionChoiceId], UserId, int]


class Md5Hash(BaseModel):
    md5: str


class Sha256Hash(BaseModel):
    sha256: str


class ImageKey(BaseModel):
    """A
    image in a dataset."""

    dataset_id: DatasetId
    image_id: ImageId


class Image(ImageKey):
    path: Path
    hash_kwargs: Md5Hash | Sha256Hash | None
    image_info: NonEmpty[dict[str, str]]
    image_group: NonEmpty[tuple[str]]


class Dataset(RootModel[list[Image]]):
    """A list of images in a dataset."""


class SelectionTaskKey(BaseModel):
    """A group of images to be presented to a user for selection."""

    images: tuple[ImageKey, ...]


class SelectionTask(SelectionTaskKey):
    difficulty: Annotated[float, Ge(0.0), Le(1.0)] = 0.5


class SelectionChoiceKey(BaseModel):
    """A user's selection of an image from a selection task."""

    images: NonEmpty[tuple[ImageKey, ...]]
    user: UserId


class Choice(BaseModel):
    index: int
    explanation: str | None = None


class SelectionChoice(SelectionChoiceKey, Choice):
    user_kind: Literal["ai", "human"]
    annotations: Sequence[dict[str, Any]]

    @model_validator(mode="after")
    def _annotations_or_explanation(self) -> Self:
        if not self.annotations and not self.explanation:
            raise ValueError(
                "At least one annotation or an explanation must be provided."
            )
        return self


class RatingTaskKey(BaseModel):
    """A group of selection choices to be presented to a user for rating."""

    choices: NonEmpty[Sequence[SelectionChoiceKey]]


class RatingTask(RatingTaskKey):
    @model_validator(mode="after")
    def _from_same_task(self) -> Self:
        """
        Ensure that all selection choices are from the same selection task.
        """
        assert all(c.images == self.images for c in self.choices)
        return self

    @model_validator(mode="after")
    def _unique_choices(self) -> Self:
        choice_keys = [choice.model_dump_json() for choice in self.choices]
        if len(set(choice_keys)) != len(choice_keys):
            raise ValueError("All choices must be unique.")

        return self

    @property
    def images(self):
        return self.choices[0].images


class RatingEvalKey(RatingTaskKey):
    """A user's evaluations of a rating task."""

    user: UserId


class RatingEval(RatingEvalKey):
    most_convincing: Choice
    most_likely_ai: Choice | None = None

    @model_validator(mode="after")
    def _most_convincing_is_valid(self) -> Self:
        if self.most_convincing.index < 0 or self.most_convincing.index >= len(
            self.choices
        ):
            raise ValueError("Most convincing index is out of bounds.")

        return self

    @model_validator(mode="after")
    def _most_likely_ai_is_valid(self) -> Self:
        if self.most_likely_ai and (
            self.most_likely_ai.index < 0
            or self.most_likely_ai.index >= len(self.choices)
        ):
            raise ValueError("Most likely AI index is out of bounds.")

        return self


class SelectionChallenge(BaseModel):
    dataset_id: DatasetId
    challenge_id: ChallengeId
    tasks: NonEmpty[Sequence[SelectionTask]]

    @property
    def difficulty(self) -> float:
        return mean(task.difficulty for task in self.tasks)
