from pydantic import BaseModel, Field

from config.base import BaseConfig


class ModelArgs(BaseModel):
    model: str = Field("gpt-4", alias="model_name")
    temperature: float = 0
    streaming: bool = False


class ModelConfig(BaseConfig):
    chat_args: ModelArgs
    agent_args: ModelArgs
