from httpx import get as http_get, AsyncClient, Response
from typing import LiteralString, Optional, Callable, Awaitable, Dict
from pydantic.v1 import BaseModel, Field

from functions.wrapper import FunctionWrapper


class YieldQueryArgs(BaseModel):
    name: str = Field(description="Protocol or project name")


class YieldQueryResult(BaseModel):
    blockchain: Optional[str] = Field(description="Blockchain name")
    symbol: Optional[str] = Field(description="Symbol of the token")
    tvl: Optional[float] = Field(description="Total value locked in USD")
    stablecoin: bool = Field(description="Whether the protocol supports stablecoin")
    apy: float = Field(description="APY in percentage")
    apy_pct_1d: Optional[float] = Field(description="APY in percentage for 1 day")
    apy_pct_7d: Optional[float] = Field(description="APY in percentage for 7 days")
    apy_pct_30d: Optional[float] = Field(description="APY in percentage for 30 days")
    il_risk: Optional[str] = Field(description="If impermanent loss risk or not")
    predictions: Dict = Field(description="Predictions of the protocol")


class YieldQuerier(FunctionWrapper[YieldQueryArgs, YieldQueryResult]):
    """Query yield information of web3 projects from the DefiLlama platform."""

    url: str = "https://yields.llama.fi/pools"

    @classmethod
    def name(cls) -> LiteralString:
        return "yield_querier"

    @classmethod
    def description(cls) -> LiteralString:
        return "query yielding data of protocol built on Solana"

    @classmethod
    def notification(cls) -> str:
        return "\n*Query yielding data on DefiLLama...*\n"

    @staticmethod
    def _create_result(resp: Response, name: str) -> YieldQueryResult:
        if resp.status_code == 200:
            body: dict = resp.json()
            if body["status"] != "success":
                raise RuntimeError(f'failed to query yield, status: {body["status"]}')
            data: list = body["data"]
            for item in data:
                if item["project"].lower() == name.lower():
                    return YieldQueryResult(
                        blockchain=item["chain"],
                        tvl=item["tvlUsd"],
                        stablecoin=item["stablecoin"],
                        apy=item["apy"],
                        apy_pct_1d=item["apyPct1D"],
                        apy_pct_7d=item["apyPct7D"],
                        apy_pct_30d=item["apyPct30D"],
                        il_risk=item["ilRisk"],
                        predictions=item["predictions"],
                    )
            raise RuntimeError(f"yield data of protocol {name} not found")
        else:
            raise RuntimeError(f"failed to query TVL: status: {resp.status_code}, response: {resp.text}")

    @property
    def func(self) -> Optional[Callable[..., YieldQueryResult]]:
        def _query_yield(name: str) -> YieldQueryResult:
            """Query defillama information from defiLlama."""
            resp = http_get(self.url)
            return self._create_result(resp, name)

        return _query_yield

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[YieldQueryResult]]]:
        async def _query_yield(name: str) -> YieldQueryResult:
            """Query defillama information from defiLlama."""
            async with AsyncClient() as client:
                resp = await client.get(self.url)
                return self._create_result(resp, name)

        return _query_yield
