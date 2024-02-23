from typing import List, Optional, LiteralString, Callable
from pydantic.v1 import BaseModel, Field

from config.chain import ChainConfig
from functions.wrapper import FunctionWrapper


class ListingArgs(BaseModel):
    limit: int = Field(20, description="The maximum number of tokens to list")


class ListingResult(BaseModel):
    tokens: List[str] = Field(description="The list of token symbols")


class TokenLister(FunctionWrapper[ListingArgs, ListingResult]):
    chain_config: ChainConfig

    def __init__(self, *, chain_config: ChainConfig):
        self.chain_config = chain_config

        super().__init__()

    @classmethod
    def name(cls) -> LiteralString:
        return "list_tokens"

    @classmethod
    def description(cls) -> LiteralString:
        return "useful for when you want to list reliable tokens on Solana"

    @classmethod
    def notification(cls) -> str:
        return "\n*Get reliable token list...*\n"

    @property
    def func(self) -> Optional[Callable[..., ListingResult]]:
        def _token_list(*, limit: int = 20) -> ListingResult:
            if limit <= 0:
                raise ValueError("limit must be a positive integer")

            if len(self.chain_config.tokens) > limit:
                tokens = self.chain_config.tokens[:limit]
            else:
                tokens = self.chain_config.tokens
            return ListingResult(tokens=[token.symbol for token in tokens])

        return _token_list
