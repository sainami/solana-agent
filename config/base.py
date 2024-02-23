from pathlib import Path
from pydantic import BaseModel


class BaseConfig(BaseModel):

    @classmethod
    def from_file(cls, path: Path) -> "BaseConfig":
        data = path.read_bytes()
        return cls.model_validate_json(data)
