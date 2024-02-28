from typing import Optional
from pydantic.v1 import BaseModel


class BaseResponse(BaseModel):
    status: bool
    error: Optional[str] = None
