from typing import Optional, List, Any, Dict
from uuid import UUID

from common.callbacks.stream_handler import StreamingCallbackHandler


class ToolCallbackHandler(StreamingCallbackHandler):

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if metadata is not None:
            notification = metadata.get("notification", None)
            if notification is not None:
                if not isinstance(notification, str):
                    notification = str(notification)
                await self.send_content(notification)
