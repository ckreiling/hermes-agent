"""A cancelled transport's cleanup must not replace discovery's timeout with cancellation."""
import asyncio

import pytest

from tools.mcp_tool import MCPServerTask


@pytest.mark.asyncio
async def test_discovery_timeout_survives_cancelled_transport_cleanup(monkeypatch):
    from tools.mcp_tool_discovery import _discover_and_register_server

    cleanup = asyncio.Event()
    shutdown = MCPServerTask.shutdown
    claimed = []

    async def run(server, config):
        claimed.append(server)
        try:
            await asyncio.Event().wait()
        finally:
            # Keep the cancelled transport unwinding until discovery claims its cleanup.
            await cleanup.wait()

    async def release_and_shutdown(server):
        cleanup.set()
        await shutdown(server)

    monkeypatch.setattr(MCPServerTask, 'run', run)
    monkeypatch.setattr(MCPServerTask, 'shutdown', release_and_shutdown)
    with pytest.raises(TimeoutError):
        await _discover_and_register_server('timeout-cleanup', {'connect_timeout': .05})
    assert len(claimed) == 1
    assert claimed[0]._task.cancelled()
    assert claimed[0].session is None


@pytest.mark.asyncio
async def test_shutdown_still_propagates_caller_cancellation(monkeypatch):
    entered = asyncio.Event()
    closing = asyncio.Event()
    server = MCPServerTask('caller-cancel')

    async def run():
        entered.set()
        await asyncio.Event().wait()

    async def closing_wait_for(awaitable, *, timeout):
        closing.set()
        return await awaitable

    server._task = asyncio.create_task(run())
    await entered.wait()
    monkeypatch.setattr(asyncio, 'wait_for', closing_wait_for)
    caller = asyncio.create_task(server.shutdown())
    await closing.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert server._task.cancelled()


@pytest.mark.asyncio
async def test_discovery_still_propagates_caller_cancellation(monkeypatch):
    from tools.mcp_tool_discovery import _discover_and_register_server

    entered = asyncio.Event()
    claimed = []

    async def run(server, config):
        claimed.append(server)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(MCPServerTask, 'run', run)
    caller = asyncio.create_task(_discover_and_register_server(
        'cancel-cleanup', {'connect_timeout': 30}))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert claimed[0]._task.cancelled()
