"""data models for diff-detect.

data flow:
    Dataset -> SelectionChallenge -> SelectionTask -> SelectionChoice -> RatingTask -> RatingEval"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Annotated, Any, Generic, Literal, Sequence, Sized, TypeVar

import streamlit as st
from annotated_types import MinLen
from pydantic import BaseModel, model_validator
from sqlmodel import JSON, Column, Field, SQLModel, create_engine
from typing_extensions import Self

SQLModel.__table_args__ = {"extend_existing": True}

S = TypeVar("S", bound=Sized)
NonEmpty = Annotated[S, MinLen(1)]

# _Id = Annotated[
#     str,
#     Predicate(lambda s: bool(s) and all(c.isalnum() or c in "_-" for c in s)),
# ]

DatasetId = Literal["butterfly"]
ImageId = str
RateTaskId = str
UserId = str
ExplainChallengeId = Literal[
    "explain_dummy", "explain_butterfly_easy", "explain_butterfly_difficult"
]
EXPLAIN_CHALLENGE_IDS: Sequence[ExplainChallengeId] = (
    "explain_dummy",
    "explain_butterfly_easy",
    "explain_butterfly_difficult",
)
RateChallengeId = Literal[
    "rate_dummy", "rate_butterfly_easy", "rate_butterfly_difficult"
]
RATE_CHALLENGE_IDS: Sequence[RateChallengeId] = (
    "rate_dummy",
    "rate_butterfly_easy",
    "rate_butterfly_difficult",
)
ChallengeId = ExplainChallengeId | RateChallengeId


class Image(SQLModel, table=True):
    """An image in a dataset."""

    id: ImageId = Field(primary_key=True)
    image_info: dict[str, Any] = Field(sa_type=JSON)
    image_group: str
    source: str


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
    hashed_password: str


_UserId = Annotated[UserId, Field(foreign_key="user.id")]


# class ReferenceImage(SQLModel):
#     image: ImageId = Field(foreign_key="image.id", primary_key=True)
#     explained_difference: ExplainedDifferenceId = Field(
#         foreign_key="explaineddifference.id", primary_key=True
#     )

TaskKey = tuple[ImageId, ImageId, ImageId]


class ExplainTask(SQLModel):
    annotated_image: ImageId = Field(foreign_key="image.id", primary_key=True)
    reference_image1: ImageId = Field(foreign_key="image.id", primary_key=True)
    reference_image2: ImageId = Field(foreign_key="image.id", primary_key=True)

    @model_validator(mode="before")
    def _order_references(cls, values: dict[str, Any]) -> dict[str, Any]:
        ref1, ref2 = sorted([values["reference_image1"], values["reference_image2"]])
        values["reference_image1"] = ref1
        values["reference_image2"] = ref2
        return values

    @property
    def task_key(self) -> TaskKey:
        return (self.annotated_image, self.reference_image1, self.reference_image2)


class ExplainOutcome(ExplainTask, table=True):
    user: UserId = Field(foreign_key="user.id", primary_key=True)
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


class RateTask(ExplainTask):
    own: UserId = Field(foreign_key="user.id", primary_key=True)
    peer: UserId = Field(foreign_key="user.id", primary_key=True)
    ai: UserId = Field(foreign_key="user.id", primary_key=True)


class RateOutcome(RateTask, table=True):
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    most_convincing: _UserId
    most_likely_ai: _UserId

    @model_validator(mode="after")
    def _valid_choices(self) -> Self:
        if self.most_convincing not in {self.own, self.peer, self.ai}:
            raise ValueError("Most convincing user must be one of the three users.")
        if self.most_likely_ai not in {self.own, self.peer, self.ai}:
            raise ValueError("Most likely AI user must be one of the three users.")
        return self


TaskT = TypeVar("TaskT", bound=ExplainTask | RateTask)


@dataclass
class _ChallengeBase(Generic[TaskT]):
    tasks: NonEmpty[list[TaskT]]

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def done_count(self) -> int:
        return len([t for t in self.tasks if isinstance(t, ExplainOutcome)])

    @property
    def finished(self) -> bool:
        return all(isinstance(t, ExplainOutcome) for t in self.tasks)

    @property
    def progress(self) -> float:
        return self.done_count / self.task_count

    @property
    def first_undone(self) -> int | None:
        for idx, task in enumerate(self.tasks):
            if not isinstance(task, ExplainOutcome):
                return idx

        return None


@dataclass
class ExplainChallenge(_ChallengeBase[ExplainTask | ExplainOutcome]):
    id: ExplainChallengeId


@dataclass
class RateChallenge(_ChallengeBase[RateTask | RateOutcome]):
    id: RateChallengeId


@dataclass
class ChallengeData:
    datasets: dict[DatasetId, Dataset]
    explain_challenges: dict[ExplainChallengeId, ExplainChallenge]
    reference_explain_outcomes: dict[tuple[TaskKey, UserId], ExplainOutcome]
    rate_challenges: dict[RateChallengeId, RateChallenge]


@dataclass
class ActiveTask:
    challenge_data: ChallengeData
    challenge_id: ExplainChallengeId | RateChallengeId
    task_idx: int


DEFAULT_SQLITE_URL = "sqlite:///database.db"
SQLITE_URL = os.environ.get("sqlite_url", DEFAULT_SQLITE_URL)


@st.cache_resource(show_spinner="Creating SQLite engine...", show_time=True)
def get_sqlite_engine():
    engine = create_engine(SQLITE_URL, echo=True)
    SQLModel.metadata.create_all(engine)
    return engine


if __name__ == "__main__":
    _ = get_sqlite_engine()
