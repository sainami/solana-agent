from pydantic import BaseModel, Field

from config.base import BaseConfig


class ModelArgs(BaseModel):
    azure_deployment: str = "web3-agent"
    temperature: float = 0
    streaming: bool = True


class ModelConfig(BaseConfig):
    chat_args: ModelArgs
    agent_args: ModelArgs
