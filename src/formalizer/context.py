import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest


@dataclass(frozen=True, slots=True)
class ContextPolicyInput:
    """Immutable input supplied to one context-policy decision."""

    messages: tuple[ModelMessage, ...]
    run_id: str
    run_step: int
    request_index: int


@dataclass(frozen=True, slots=True)
class ContextProjection:
    """The history a context policy chooses to expose to the next model request."""

    messages: Sequence[ModelMessage]
    action: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


class ContextPolicy(Protocol):
    """Project the active agent history into the history visible to one model request."""

    @property
    def name(self) -> str: ...

    async def project(self, context: ContextPolicyInput) -> ContextProjection: ...


class InvalidContextProjection(ValueError):
    """A context policy returned history that cannot be sent to a model."""


class ContextController:
    """Apply a context policy and durably audit every attempted projection."""

    def __init__(
        self,
        *,
        policy: ContextPolicy | None = None,
        events_path: Path | None = None,
    ) -> None:
        if (policy is None) != (events_path is None):
            raise ValueError("policy and events_path must either both be set or both be omitted")
        self._policy = policy
        self._events_path = events_path
        self._request_index = 0

    async def process(
        self,
        messages: list[ModelMessage],
        *,
        run_id: str,
        run_step: int,
    ) -> list[ModelMessage]:
        if self._policy is None:
            return messages

        request_index = self._request_index
        self._request_index += 1
        before_json = ModelMessagesTypeAdapter.dump_json(messages)
        self._append(
            {
                "schema_version": 1,
                "event": "context_projection_started",
                "recorded_at": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "run_step": run_step,
                "request_index": request_index,
                "policy": self._policy.name,
                "before_message_count": len(messages),
                "before_sha256": sha256(before_json).hexdigest(),
                "before_messages": json.loads(before_json),
            }
        )

        try:
            projection = await self._policy.project(
                ContextPolicyInput(
                    messages=tuple(deepcopy(messages)),
                    run_id=run_id,
                    run_step=run_step,
                    request_index=request_index,
                )
            )
            projected_messages = list(projection.messages)
            if not projected_messages:
                raise InvalidContextProjection("projected history cannot be empty")
            if not isinstance(projected_messages[-1], ModelRequest):
                raise InvalidContextProjection("projected history must end with a ModelRequest")
            if not projection.action.strip():
                raise InvalidContextProjection("projection action cannot be blank")
            after_json = ModelMessagesTypeAdapter.dump_json(projected_messages)
            metadata_json = json.loads(json.dumps(dict(projection.metadata)))
        except BaseException as error:
            self._append(
                {
                    "schema_version": 1,
                    "event": "context_projection_failed",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "run_id": run_id,
                    "run_step": run_step,
                    "request_index": request_index,
                    "policy": self._policy.name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise

        self._append(
            {
                "schema_version": 1,
                "event": "context_projection_completed",
                "recorded_at": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "run_step": run_step,
                "request_index": request_index,
                "policy": self._policy.name,
                "action": projection.action,
                "after_message_count": len(projected_messages),
                "after_sha256": sha256(after_json).hexdigest(),
                "after_messages": json.loads(after_json),
                "metadata": metadata_json,
            }
        )
        return projected_messages

    def _append(self, record: Mapping[str, object]) -> None:
        assert self._events_path is not None
        encoded = f"{json.dumps(record, sort_keys=True, separators=(',', ':'))}\n"
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
