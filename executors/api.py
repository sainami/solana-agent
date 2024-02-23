import asyncio

from typing import List, Annotated, Tuple
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from common.callbacks.tool_handler import ToolCallbackHandler
from executors.chatter import Chatter


def register_agent_api(chatter: Chatter) -> APIRouter:
    router = APIRouter(prefix="/api/agent", tags=["Web3 Agent Chatting API Endpoints"])

    @router.post("/chat")
    async def chat(
        question: Annotated[str, Body()],
        chat_history: List[Tuple[str, str]],
    ):
        # TODO: use a callback manager is better
        callback_handler = ToolCallbackHandler()

        async def _chatting():
            try:
                await chatter.chat(
                    question,
                    chat_history,
                    config={
                        "callbacks": [callback_handler]
                    }
                )
            except Exception as e:
                await callback_handler.send_error(repr(e))
            finally:
                await callback_handler.stop()

        _ = asyncio.create_task(_chatting())
        return StreamingResponse(
            callback_handler,
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )

    return router
