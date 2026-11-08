"""Turn-scoped presentation identity, never an execution or approval authority.

The mutable active bit invalidates copied worker / persistent-kernel contexts when
an operation completes. Identity is scoped to its owning TurnRunner, not a name
FIFO or the last tool started in a session.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock


@dataclass
class _Operation:
    owner: object
    payload: dict
    active: bool = True


_current: ContextVar[_Operation | None] = ContextVar('gateway_tool_operation', default=None)


class ToolOperations:
    def __init__(self):
        self._lock = RLock()
        self._open = {}
        self._active = True

    def bind(self, call_id, name, args):
        from agent.display import redact_tool_args_for_display
        from agent.redact import redact_sensitive_text
        from gateway.session_tool_events import bounded_args
        import json

        args = args if isinstance(args, dict) else {}
        safe_args = json.loads(redact_sensitive_text(json.dumps(
            redact_tool_args_for_display(name, args), ensure_ascii=True, default=str), force=True))
        payload = {'tool_call_id': str(call_id), 'name': str(name), 'args': bounded_args(safe_args)}
        with self._lock:
            if not self._active:
                return
            previous = self._open.get(str(call_id))
            if previous is not None:
                previous.active = False
            operation = _Operation(self, payload)
            self._open[str(call_id)] = operation
            _current.set(operation)

    def presentation(self):
        with self._lock:
            operation = _current.get()
            if self._active and operation is not None and operation.owner is self and operation.active:
                return dict(operation.payload)
        return None

    def retire(self, call_id):
        with self._lock:
            operation = self._open.pop(str(call_id), None)
            if operation is not None:
                operation.active = False

    def close(self):
        with self._lock:
            self._active = False
            for operation in self._open.values():
                operation.active = False
            self._open.clear()
