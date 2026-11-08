"""A replacement ACP process joins a still-running original operation."""
import asyncio
import json
import shlex

import pytest

from tests.acp_adapter.test_gateway_fork_wire import answer, initialize, model_peer, new, permissions, prompt, updates
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.waiting import async_wait_until


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_default_pending_approval_reattach_keeps_original_card_identity(daemon, tmp_path, model_peer):
    target = tmp_path / 'owned-dir'
    target.mkdir()
    (target / 'owned').write_text('disposable')
    model_peer.provider.push(ToolCall('terminal', {'command': 'rm -r -- ' + shlex.quote(str(target))}),
                             Text('REATTACHED_FINAL'))
    async with viewer(daemon) as ws:
        async with editor(daemon, tmp_path) as first:
            await initialize(first)
            sid = await new(first, tmp_path)
            task = asyncio.create_task(prompt(first, sid))
            try:
                await async_wait_until(lambda: bool(permissions(first)), timeout=20, what='first approval')
                first_permission = permissions(first)[0]
                original_id = first_permission['params']['toolCall']['toolCallId']
                snapshot = await ws.rpc('session.resume', session_id=sid)
                assert snapshot['prompts'][0]['operation']['tool_call_id'] == original_id
                assert target.exists()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert target.exists(), 'detach is neither approval nor denial'
        async with editor(daemon, tmp_path) as second:
            await initialize(second)
            loaded = await second.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
            assert 'result' in loaded, loaded
            await async_wait_until(lambda: bool(permissions(second)), timeout=20, what='reoffered approval')
            permission = permissions(second)[0]
            assert permission['params']['toolCall']['toolCallId'] == original_id
            own = [u for u in updates(second) if u.get('toolCallId') == original_id]
            assert [u['sessionUpdate'] for u in own] == ['tool_call'], own
            await answer(second, permission, 'allow_once')
            await async_wait_until(lambda: 'REATTACHED_FINAL' in ''.join(
                u['content']['text'] for u in updates(second) if u['sessionUpdate'] == 'agent_message_chunk'),
                timeout=25, what='original operation final')
            assert not target.exists()
            own = [u for u in updates(second) if u.get('toolCallId') == original_id]
            assert [u['sessionUpdate'] for u in own] == ['tool_call', 'tool_call_update'], own
            print('FORK_REATTACH_RECEIPT=' + json.dumps({'session_id': sid, 'operation': original_id,
                                                       'card_updates': own}))
