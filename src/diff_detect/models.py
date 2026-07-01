"""data models for diff-detect.

data flow:
    Dataset -> SelectionChallenge -> SelectionTask -> SelectionChoice -> RatingTask -> RatingEval"""

from __future__ import annotations

import datetime
from enum import StrEnum, auto
from typing import Annotated, Any, Literal, NewType, Sequence, Sized, TypeVar

from annotated_types import MinLen
from pydantic import BaseModel, model_validator
from sqlmodel import JSON, Column, Field, SQLModel, create_engine
from typing_extensions import Self

S = TypeVar("S", bound=Sized)
NonEmpty = Annotated[S, MinLen(1)]

# _Id = Annotated[
#     str,
#     Predicate(lambda s: bool(s) and all(c.isalnum() or c in "_-" for c in s)),
# ]

UserId = str
ExplainedDifferenceId = NewType("ExplainedDifferenceId", int)
DatasetId = Literal["butterfly"]
ImageId = str
TaskId = NewType("TaskId", str)
ExplainChallengeId = Literal[
    "select_dummy", "select_butterfly_easy", "select_butterfly_difficult"
]
RateChallengeId = Literal[
    "rate_dummy", "rate_butterfly_easy", "rate_butterfly_difficult"
]
ChallengeId = ExplainChallengeId | RateChallengeId


class Image(SQLModel, table=True):
    """An image in a dataset."""

    id: ImageId = Field(primary_key=True)
    image_info: dict[str, Any] = Field(sa_type=JSON)
    image_group: str
    source: str


_ImageId = Annotated[ImageId, Field(foreign_key="image.id")]


# class LocalImage(Image):
#     path: Path


class Dataset(BaseModel):
    """A list of images in a dataset."""

    # dataset_id: DatasetId
    images: dict[ImageId, Image]

    def __getitem__(self, key: ImageId) -> Image:
        return self.images[key]


class UserRole(StrEnum):
    MAINTAINER = auto()
    PARTICIPANT = auto()


class UserKind(StrEnum):
    HUMAN = auto()
    AI = auto()


class User(SQLModel, table=True):
    id: UserId = Field(primary_key=True)
    kind: UserKind
    role: UserRole


_UserId = Annotated[UserId, Field(foreign_key="user.id")]


# class ReferenceImage(SQLModel):
#     image: ImageId = Field(foreign_key="image.id", primary_key=True)
#     explained_difference: ExplainedDifferenceId = Field(
#         foreign_key="explaineddifference.id", primary_key=True
#     )


class ExplainDiffernceTask(SQLModel):
    annotated_image: ImageId
    reference_image1: ImageId
    reference_image2: ImageId

    @model_validator(mode="before")
    def _order_references(cls, values: dict[str, Any]) -> dict[str, Any]:
        ref1, ref2 = sorted([values["reference_image1"], values["reference_image2"]])
        values["reference_image1"] = ref1
        values["reference_image2"] = ref2
        return values


class ExplainedDifference(ExplainDiffernceTask, table=True):
    id: ExplainedDifferenceId | None = Field(primary_key=True)
    user: _UserId
    explanation: str | None
    annotations: dict[str, Any] | None = Field(sa_column=Column(JSON))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @model_validator(mode="after")
    def _annotations_or_explanation(self) -> Self:
        if not self.annotations and not self.explanation:
            raise ValueError(
                "At least one annotation or an explanation must be provided."
            )
        return self


_ExplainedDifferenceId = Annotated[
    ExplainedDifferenceId, Field(foreign_key="explaineddifference.id")
]


class ExplanationRating(SQLModel, table=True):
    id: int | None = Field(primary_key=True)
    self: _ExplainedDifferenceId
    peer: _ExplainedDifferenceId
    ai: _ExplainedDifferenceId
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @model_validator(mode="after")
    def _unique_users(self) -> Self:
        if len({self.self, self.peer, self.ai}) != 3:
            raise ValueError("All users must be different.")
        return self

    most_convincing: _ExplainedDifferenceId
    most_likely_ai: _ExplainedDifferenceId

    @model_validator(mode="after")
    def _valid_choices(self) -> Self:
        if self.most_convincing not in {self.self, self.peer, self.ai}:
            raise ValueError("Most convincing user must be one of the three users.")
        if self.most_likely_ai not in {self.self, self.peer, self.ai}:
            raise ValueError("Most likely AI user must be one of the three users.")
        return self


class ExplainDifferenceChallenge(BaseModel):
    challenge_id: ExplainChallengeId
    tasks: NonEmpty[Sequence[ExplainDiffernceTask]]


class RateTask(BaseModel):
    self: ExplainedDifferenceId
    peer: ExplainedDifferenceId
    ai: ExplainedDifferenceId


class RateChallenge(BaseModel):
    challenge_id: RateChallengeId
    tasks: NonEmpty[Sequence[RateTask]]


if __name__ == "__main__":
    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)

    SQLModel.metadata.create_all(engine)
