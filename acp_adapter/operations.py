"""Turn-scoped ACP presentation identity; never an execution/approval authority.

Worker context propagation (including execute_code's CellAuthority) carries a
reference to the live operation. Retiring that shared record also invalidates
copies of its context, without changing the executor's callback contract.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

import acp


@dataclass
class _Operation:
    owner: ToolCallOperations
    start: Any
    standalone: bool = False
    active: bool = True


_current_operation: ContextVar[_Operation | None] = ContextVar("acp_operation", default=None)


class ToolCallOperations:
    """Correlate permission requests with operations opened by this ACP turn."""

    def __init__(self, send_update: Callable[[Any], None]):
        self._send = send_update
        self._open: dict[str, _Operation] = {}
        self._lock = RLock()
        self._active = True

    def bind(self, start: Any) -> None:
        """Called synchronously in the executing worker, before nested dispatch."""
        with self._lock:
            if not self._active:
                return
            operation = _Operation(self, start)
            self._open[start.tool_call_id] = operation
            _current_operation.set(operation)

    def retire(self, tool_call_id: str) -> None:
        with self._lock:
            if operation := self._open.pop(tool_call_id, None):
                operation.active = False

    def close(self) -> None:
        """Invalidate copied contexts and close standalone prompts on turn teardown."""
        with self._lock:
            self._active = False
            for operation in list(self._open.values()):
                operation.active = False
                if operation.standalone:
                    self._send(acp.update_tool_call(
                        operation.start.tool_call_id, title=operation.start.title,
                        kind=operation.start.kind, status="failed",
                    ))
            self._open.clear()

    def request_permission(self, request_permission_fn, loop, session_id, *, tool_call, options, timeout, what):
        """Use the live operation, or explicitly open/close a standalone approval.

        Permission resolution is not tool completion: a script may catch a
        denied nested edit and continue. Only standalone approvals end here.
        """
        from .permissions import await_permission

        standalone = None
        with self._lock:
            if not self._active:
                return None, False
            operation = _current_operation.get()
            if operation is not None and operation.owner is self and operation.active:
                tool_call = tool_call.model_copy(update={
                    "tool_call_id": operation.start.tool_call_id, "title": operation.start.title,
                })
            else:
                start = acp.start_tool_call(
                    tool_call.tool_call_id, tool_call.title or "Permission request",
                    kind=tool_call.kind or "other", status="pending",
                    content=tool_call.content, raw_input=tool_call.raw_input,
                    locations=tool_call.locations,
                )
                standalone = _Operation(self, start, standalone=True)
                self._open[start.tool_call_id] = standalone
                self._send(start)
        allowed = False
        try:
            response, timed_out = await_permission(
                request_permission_fn, loop, session_id, tool_call=tool_call,
                options=options, timeout=timeout, what=what,
            )
            outcome = getattr(response, "outcome", None)
            allowed = getattr(outcome, "outcome", None) == "selected" and any(
                option.option_id == getattr(outcome, "option_id", None) and option.kind.startswith("allow")
                for option in options
            )
            return response, timed_out
        finally:
            if standalone is not None:
                with self._lock:
                    if standalone.active:
                        self.retire(standalone.start.tool_call_id)
                        self._send(acp.update_tool_call(
                            standalone.start.tool_call_id, title=standalone.start.title,
                            kind=standalone.start.kind, status="completed" if allowed else "failed",
                        ))
