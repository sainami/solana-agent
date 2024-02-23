from asyncio import Queue
from typing import Any, Optional, List
from uuid import UUID
from pydantic import BaseModel
from langchain.callbacks.base import AsyncCallbackHandler

from common.response import BaseResponse


class StreamingResponse(BaseResponse):
    content: Optional[str] = None
    metadata: Optional[BaseModel] = None


class StreamingCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self._queue = Queue()

    def __aiter__(self):
        return self

    @staticmethod
    def _frame(data: StreamingResponse) -> str:
        return "data: " + data.model_dump_json() + "\n\n"

    async def __anext__(self):
        data = await self._queue.get()
        if isinstance(data, StreamingResponse):
            return self._frame(data)
        else:
            raise StopAsyncIteration

    async def send_content(self, data: str):
        await self._queue.put(StreamingResponse(status=True, content=data))

    async def send_error(self, err: str):
        await self._queue.put(StreamingResponse(status=False, error=err))

    async def send_metadata(self, metadata: BaseModel):
        await self._queue.put(StreamingResponse(status=True, metadata=metadata))

    async def stop(self):
        await self._queue.put(None)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        await self._queue.put(StreamingResponse(status=True, content=token))

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        await self._queue.put(StreamingResponse(status=False, error=f"LLM error: {error}"))
