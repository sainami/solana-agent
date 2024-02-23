from httpx import get as http_get, AsyncClient, Response
from typing import LiteralString, List, Dict, Optional, Callable, Awaitable
from pydantic.v1 import BaseModel, Field

from functions.wrapper import FunctionWrapper


class PriceQueryArgs(BaseModel):
    base_tokens: List[str] = Field(description="Symbol of all the base tokens")
    quote_token: str = Field("USDC", description="Symbol of the quote token")


class PriceResult(BaseModel):
    prices: Dict[str, float] = Field(description="Price of the base tokens in the quote token")


class PriceQuerier(FunctionWrapper[PriceQueryArgs, PriceResult]):
    """Query price information from the jupiter service."""

    base_url: str = "https://price.jup.ag/v4"

    @classmethod
    def name(cls) -> LiteralString:
        return "get_price"

    @classmethod
    def description(cls) -> LiteralString:
        return "find the best price of tokens on Solana DEXes"

    @classmethod
    def notification(cls) -> str:
        return "\n*Query token price from Jupiter...*\n"

    @staticmethod
    def _create_result(resp: Response) -> PriceResult:
        if resp.status_code == 200:
            data: Dict[str, Dict] = resp.json()["data"]
            return PriceResult(prices={k: v["price"] for k, v in data.items()})
        else:
            raise RuntimeError(f"failed to query price: status: {resp.status_code}, response: {resp.text}")

    @property
    def func(self) -> Optional[Callable[..., PriceResult]]:
        def _get_price(base_tokens: List[str], quote_token: str = "USDC") -> PriceResult:
            resp = http_get(
                self.base_url + "/price",
                params={
                    "ids": base_tokens,
                    "vsToken": quote_token,
                }
            )
            return self._create_result(resp)

        return _get_price

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[PriceResult]]]:
        async def _get_price(base_tokens: List[str], quote_token: str = "USDC") -> PriceResult:
            async with AsyncClient() as client:
                resp = await client.get(
                    self.base_url + "/price",
                    params={
                        "ids": base_tokens,
                        "vsToken": quote_token,
                    }
                )
                return self._create_result(resp)

        return _get_price
