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
    TypeAdapter,
    model_validator,
)
from pydantic import (
    Field as PydanticField,
)
from sqlalchemy import TypeDecorator
from sqlmodel import JSON, Column, Field, SQLModel, create_engine
from typing_extensions import Self

SQLModel.__table_args__ = {"extend_existing": True}

S = TypeVar("S", bound=Sized)
NonEmpty = Annotated[S, MinLen(1)]


class DatasetId(StrEnum):
    BUTTERFLY = auto()
    FLYBUTTER = auto()


ImageId = str
RateTaskId = str
UserId = str
ExplainChallengeId = Literal[
    "explain_dummy",
    "explain_butterfly_easy",
    "explain_butterfly_difficult",
    "explain_flybutter_easy",
]
EXPLAIN_CHALLENGE_IDS: Sequence[ExplainChallengeId] = (
    "explain_dummy",
    "explain_butterfly_easy",
    "explain_butterfly_difficult",
    "explain_flybutter_easy",
)
RateChallengeId = Literal[
    "rate_dummy",
    "rate_butterfly_easy",
    "rate_butterfly_difficult",
    "rate_flybutter_easy",
]
RATE_CHALLENGE_IDS: Sequence[RateChallengeId] = (
    "rate_dummy",
    "rate_butterfly_easy",
    "rate_butterfly_difficult",
    "rate_flybutter_easy",
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
    name: str
    lab: str | None
    kind: UserKind
    role: UserRole
    hashed_password: str | None


_UserId = Annotated[UserId, Field(foreign_key="user.id")]


TaskKey = tuple[ImageId, ImageId, ImageId]
SelectionKey = tuple[TaskKey, int]


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
    def image_ids(self) -> TaskKey:
        return (self.annotated_image, self.reference_image1, self.reference_image2)

    @property
    def task_key(self) -> TaskKey:
        return self.image_ids

    @property
    def candidate_key(self) -> TaskKey:
        return cast(TaskKey, tuple(sorted(self.image_ids)))

    @property
    def selection_index(self) -> int:
        return self.candidate_key.index(self.annotated_image)

    @property
    def selection_key(self) -> SelectionKey:
        return self.candidate_key, self.selection_index

    def references_for(self, annotated_image: ImageId) -> tuple[ImageId, ImageId]:
        references = tuple(
            image_id for image_id in self.image_ids if image_id != annotated_image
        )
        if len(references) != 2:
            raise ValueError("Annotated image must be one of the task images.")
        return cast(tuple[ImageId, ImageId], references)


class CanvasObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = ""
    stroke: str | None = None


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

    def as_explain_task(self) -> ExplainTask:
        return ExplainTask(
            dataset_id=self.dataset_id,
            annotated_image=self.annotated_image,
            reference_image1=self.reference_image1,
            reference_image2=self.reference_image2,
        )


class RateTask(ExplainTask):
    own: UserId = Field(foreign_key="user.id", primary_key=True)
    peer: UserId = Field(foreign_key="user.id", primary_key=True)
    ai: UserId = Field(foreign_key="user.id", primary_key=True)


class RateOutcome(RateTask, table=True):
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    most_convincing: _UserId | None = None
    most_likely_ai: _UserId | None = None

    @property
    def complete(self) -> bool:
        return self.most_convincing is not None and self.most_likely_ai is not None

    @model_validator(mode="after")
    def _valid_choices(self) -> Self:
        choices = {self.own, self.peer, self.ai}
        if self.most_convincing is not None and self.most_convincing not in choices:
            raise ValueError("Most convincing user must be one of the three users.")
        if self.most_likely_ai is not None and self.most_likely_ai not in choices:
            raise ValueError("Most likely AI user must be one of the three users.")
        return self


TaskT = TypeVar("TaskT", bound=ExplainTask | RateTask)


@dataclass
class _ChallengeBase(Generic[TaskT]):
    tasks: NonEmpty[list[TaskT]]

    @staticmethod
    def _task_done(task: TaskT) -> bool:
        return isinstance(task, ExplainOutcome) or (
            isinstance(task, RateOutcome) and task.complete
        )

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def done_count(self) -> int:
        return len([task for task in self.tasks if self._task_done(task)])

    @property
    def finished(self) -> bool:
        return all(self._task_done(task) for task in self.tasks)

    @property
    def progress(self) -> float:
        if self.task_count < 1:
            return 0.0
        else:
            return self.done_count / self.task_count

    @property
    def first_undone(self) -> int | None:
        for idx, task in enumerate(self.tasks):
            if not self._task_done(task):
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
    reference_explain_outcomes: dict[tuple[SelectionKey, UserId], ExplainOutcome]
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
