from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from semantic_action_frame_guard.v5_models import ToolCallSpec
from semantic_action_frame_guard.v4_models import StrictModel


class ConversationHistoryMessage(StrictModel):
    """The Task Shield-shaped, pre-current-message conversation entry."""

    role: str = Field(min_length=1)
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    error: str | None = None


class GuardInput(StrictModel):
    task_id: str = Field(min_length=1)
    trusted_user_task: str = Field(min_length=1)
    proposed_tool_call: ToolCallSpec
    conversation_history: list[ConversationHistoryMessage] = Field(default_factory=list)
    related_content: str = ""

    @model_validator(mode="after")
    def proposed_call_is_not_historical(self) -> GuardInput:
        historical_ids: set[str] = set()
        for message in self.conversation_history:
            if message.tool_call_id:
                historical_ids.add(message.tool_call_id)
            for call in message.tool_calls or []:
                call_id = call.get("call_id")
                if isinstance(call_id, str) and call_id:
                    historical_ids.add(call_id)
        # A normal Task Shield conversation contains the same call ID once in
        # an Assistant tool call and again in its Tool result, so duplicates
        # across those paired messages are expected.
        if self.proposed_tool_call.tool_call_id in historical_ids:
            raise ValueError("the proposed call cannot already be in conversation history")
        return self


__all__ = ["ConversationHistoryMessage", "GuardInput"]
