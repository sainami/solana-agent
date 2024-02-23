from httpx import get as http_get, AsyncClient, Response
from typing import LiteralString, Optional, Callable, Awaitable
from pydantic.v1 import BaseModel, Field

from functions.wrapper import FunctionWrapper


class TVLQueryArgs(BaseModel):
    name: Optional[str] = Field(None, description="Protocol or project name")
    blockchain: Optional[str] = Field(None, description="Blockchain name")


class TVLQueryResult(BaseModel):
    tvl: float = Field(description="Total value locked in USD")


class TVLQuerier(FunctionWrapper[TVLQueryArgs, TVLQueryResult]):
    """Query TVL information of web3 projects from the DefiLlama platform."""

    base_url: str = "https://api.llama.fi"

    @classmethod
    def name(cls) -> LiteralString:
        return "tvl_querier"

    @classmethod
    def description(cls) -> LiteralString:
        return "query TVL data of protocol built on Solana"

    @classmethod
    def notification(cls) -> str:
        return "\n*Query TVL data on DefiLLama...*\n"

    @staticmethod
    def _create_result_with_protocol_without_blockchain(resp: Response) -> TVLQueryResult:
        if resp.status_code == 200:
            return TVLQueryResult(tvl=float(resp.text))
        else:
            raise RuntimeError(f"failed to query TVL: status: {resp.status_code}, response: {resp.text}")

    @staticmethod
    def _create_result_with_protocol_with_blockchain(
        resp: Response, protocol: str, blockchain: str,
    ) -> TVLQueryResult:
        if resp.status_code != 200:
            raise RuntimeError(f"failed to query TVL: status: {resp.status_code}, response: {resp.text}")

        body: list = resp.json()
        for data in body:
            if str(data["name"]).lower() == protocol.lower():
                for chain, tvl in data["chainTvls"].items():
                    if chain.lower() == blockchain.lower():
                        return TVLQueryResult(tvl=float(tvl))
                raise RuntimeError(f"TVL of protocol {protocol} in blockchain {blockchain} not found")
        raise RuntimeError(f"TVL of protocol {protocol} not found")

    @staticmethod
    def _create_result_without_protocol_with_blockchain(resp: Response, blockchain: str) -> TVLQueryResult:
        if resp.status_code != 200:
            raise RuntimeError(f"failed to query TVL: status: {resp.status_code}, response: {resp.text}")

        body: list = resp.json()
        for data in body:
            if str(data["name"]).lower() == blockchain.lower():
                return TVLQueryResult(tvl=float(data["tvl"]))
        raise RuntimeError(f"TVL of blockchain {blockchain} not found")

    @property
    def func(self) -> Optional[Callable[..., TVLQueryResult]]:
        def _query_tvl(
            name: Optional[str] = None,
            blockchain: Optional[str] = None,
        ) -> TVLQueryResult:
            """Query defillama information from defiLlama."""
            if name is not None and blockchain is not None:
                resp = http_get(self.base_url + "/protocols")
                return self._create_result_with_protocol_with_blockchain(resp, name, blockchain)
            elif name is not None and blockchain is None:
                resp = http_get(self.base_url + f"/tvl/{name}")
                return self._create_result_with_protocol_without_blockchain(resp)
            elif name is None and blockchain is not None:
                resp = http_get(self.base_url + "/v2/chains")
                return self._create_result_without_protocol_with_blockchain(resp, blockchain)
            else:
                raise ValueError("protocol or blockchain is required")

        return _query_tvl

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[TVLQueryResult]]]:
        async def _query_tvl(
            name: Optional[str] = None,
            blockchain: Optional[str] = None,
        ) -> TVLQueryResult:
            """Query defillama information from defiLlama."""
            async with AsyncClient() as client:
                if name is not None and blockchain is not None:
                    resp = await client.get(self.base_url + "/protocols")
                    return self._create_result_with_protocol_with_blockchain(resp, name, blockchain)
                elif name is not None and blockchain is None:
                    resp = await client.get(self.base_url + f"/tvl/{name}")
                    return self._create_result_with_protocol_without_blockchain(resp)
                elif name is None and blockchain is not None:
                    resp = await client.get(self.base_url + "/v2/chains")
                    return self._create_result_without_protocol_with_blockchain(resp, blockchain)
                else:
                    raise ValueError("protocol or blockchain is required")

        return _query_tvl
