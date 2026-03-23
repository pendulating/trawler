from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError


class Component(BaseModel):
    name: str
    summary: str = ""


class FeedbackIteration(BaseModel):
    iteration: int
    feedback: str
    refined: str


class DefinitionRecord(BaseModel):
    component: str
    draft: str
    refined: str | None = None
    feedback_iterations: list[FeedbackIteration] = Field(default_factory=list)


class QuestionRecord(BaseModel):
    id: str
    question: str
    rationale: str | None = None


class SampledAnswer(BaseModel):
    sample_id: int
    answer: str


class GroundedAnswer(BaseModel):
    question_id: str
    question: str
    answer_hat: str
    reasoning_trace: str | None = None


class AlignmentRecord(BaseModel):
    question_id: str
    score: float
    notes: str


def validate_model(payload: Any, model_cls: type[BaseModel], name: str) -> BaseModel:
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Validation failed for {name}: {exc}") from exc

