"""data models for diff-detect.

data flow:
    Dataset -> SelectionChallenge -> SelectionTask -> SelectionChoice -> RatingTask -> RatingEval"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Annotated, Any, Generic, Literal, Sequence, Sized, TypeVar, cast

import streamlit as st
from annotated_types import MinLen
from pydantic import (
    BaseModel,
    ConfigDict,
    Field as PydanticField,
    TypeAdapter,
    model_validator,
)
from sqlalchemy import TypeDecorator
from sqlmodel import JSON, Column, Field, SQLModel, create_engine
from typing_extensions import Self

SQLModel.__table_args__ = {"extend_existing": True}

S = TypeVar("S", bound=Sized)
NonEmpty = Annotated[S, MinLen(1)]


class DatasetId(StrEnum):
    BUTTERFLY = auto()


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


class PydanticJson(TypeDecorator):
    """Allows a pydantic model to be stored as a JSON column in SQLModel.

    Example usage:
    ```python
    class Nested(BaseModel):
        value: str

    class Parent(SQLModel, table=True):
        id: int = Field(primary_key=True, default=None)
        nested: Nested | None = Field(sa_column=Column(PydanticJson(Nested)))
        nested_list: list[Nested] = Field(sa_column=Column(PydanticJson(list[Nested])))
    ```
    """

    impl = JSON()
    cache_ok = True

    def __init__(self, pt: Any) -> None:
        super().__init__()
        self.pt = TypeAdapter(pt)

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None

        return self.pt.dump_python(value, mode="json")

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None

        return self.pt.validate_python(value)


class Image(SQLModel, table=True):
    """An image in a dataset."""

    dataset_id: DatasetId = Field(primary_key=True)
    image_id: ImageId = Field(primary_key=True)
    image_info: dict[str, Any] = Field(sa_type=JSON)
    image_group: str
    source: str


@dataclass
class Dataset:
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
    lab: str
    kind: UserKind
    role: UserRole
    hashed_password: str


_UserId = Annotated[UserId, Field(foreign_key="user.id")]


TaskKey = tuple[ImageId, ImageId, ImageId]


class ExplainTask(SQLModel):
    dataset_id: DatasetId = Field(foreign_key="image.dataset_id")
    annotated_image: ImageId = Field(foreign_key="image.image_id", primary_key=True)
    reference_image1: ImageId = Field(foreign_key="image.image_id", primary_key=True)
    reference_image2: ImageId = Field(foreign_key="image.image_id", primary_key=True)

    @model_validator(mode="before")
    def _order_references(cls, values: dict[str, Any]) -> dict[str, Any]:
        ref1, ref2 = sorted([values["reference_image1"], values["reference_image2"]])
        values["reference_image1"] = ref1
        values["reference_image2"] = ref2
        return values

    @property
    def task_key(self) -> TaskKey:
        return (self.annotated_image, self.reference_image1, self.reference_image2)


class CanvasObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = ""
    stroke: str | None = None
    fill: str | None = None


class CanvasJson(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str = ""
    objects: list[CanvasObject] = PydanticField(default_factory=list)
    background: str | None = None


class Annotation(BaseModel):
    """Persisted drawable-canvas state, excluding rendered image data."""

    model_config = ConfigDict(extra="allow")

    raw: CanvasJson

    @model_validator(mode="before")
    @classmethod
    def _remove_image_data(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "data" not in value:
            return value

        return {key: item for key, item in value.items() if key != "data"}

    @property
    def has_objects(self) -> bool:
        return bool(self.raw.objects)


class ExplainOutcome(ExplainTask, table=True):
    user: UserId = Field(foreign_key="user.id", primary_key=True)
    explanation: str | None
    annotation: Annotation | None = Field(sa_column=Column(PydanticJson(Annotation)))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @model_validator(mode="after")
    def _annotations_or_explanation(self) -> Self:
        if not self.annotation and not self.explanation:
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
        return len(
            [t for t in self.tasks if isinstance(t, (ExplainOutcome, RateOutcome))]
        )

    @property
    def finished(self) -> bool:
        return all(isinstance(t, (ExplainOutcome, RateOutcome)) for t in self.tasks)

    @property
    def progress(self) -> float:
        return self.done_count / self.task_count

    @property
    def first_undone(self) -> int | None:
        for idx, task in enumerate(self.tasks):
            if not isinstance(task, (ExplainOutcome, RateOutcome)):
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
    reference_explain_outcomes: dict[tuple[TaskKey, UserId], ExplainOutcome]
    challenges: dict[ChallengeId, ExplainChallenge | RateChallenge]

    @property
    def explain_challenges(self) -> dict[ExplainChallengeId, ExplainChallenge]:
        return {
            cast(ExplainChallengeId, k): v
            for k, v in self.challenges.items()
            if k in EXPLAIN_CHALLENGE_IDS and isinstance(v, ExplainChallenge)
        }

    @property
    def rate_challenges(self) -> dict[RateChallengeId, RateChallenge]:
        return {
            cast(RateChallengeId, k): v
            for k, v in self.challenges.items()
            if k in RATE_CHALLENGE_IDS and isinstance(v, RateChallenge)
        }


@dataclass
class ActiveExplainChallenge:
    challenge_data: ChallengeData
    challenge_id: ExplainChallengeId

    @property
    def challenge(self) -> ExplainChallenge:
        return cast(ExplainChallenge, self.challenge_data.challenges[self.challenge_id])


@dataclass
class ActiveRateChallenge:
    challenge_data: ChallengeData
    challenge_id: RateChallengeId

    @property
    def challenge(self) -> RateChallenge:
        return cast(RateChallenge, self.challenge_data.challenges[self.challenge_id])


DEFAULT_SQLITE_URL = "sqlite:///database.db"
SQLITE_URL = os.environ.get("sqlite_url", DEFAULT_SQLITE_URL)


@st.cache_resource(show_spinner="Creating SQLite engine...", show_time=True)
def get_sqlite_engine():
    engine = create_engine(SQLITE_URL, echo=True)
    SQLModel.metadata.create_all(engine)
    return engine


if __name__ == "__main__":
    _ = get_sqlite_engine()
