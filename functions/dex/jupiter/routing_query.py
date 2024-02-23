from httpx import get as http_get, AsyncClient, Response
from typing import Union, Literal, LiteralString, List, Mapping, Optional, Callable, Awaitable
from pydantic.v1 import BaseModel, Field

from functions.wrapper import FunctionWrapper
from config.chain import ChainConfig, TokenMetadata


class RoutingQueryArgs(BaseModel):
    amount: float = Field(description="Amount of the token to swap in or swap out")
    swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]] = Field(description="Type of the swap event")
    token_in_symbol: str = Field(description="Symbol of the token to swap in")
    token_out_symbol: str = Field(description="Symbol of the token to swap out")
    slippage_bps: float = Field(0.5, description="Slippage percentage tolerance")


class RoutingResult(BaseModel):
    amount_in: float = Field(description="Amount of the token to swap in")
    amount_out: float = Field(description="Amount of the token to swap out")
    price_impact_pct: float = Field(description="Price impact percentage")
    route_plan: List["Route"] = Field(description="Swap route plan")

    class Route(BaseModel):
        swap_info: "SwapInfo" = Field(description="Swap information")
        percent: int = Field(description="Percentage of the swap")

        class SwapInfo(BaseModel):
            amm_address: str = Field(description="Address of the AMM pool")
            label: str = Field(description="Label of the AMM pool")
            token_in_symbol: str = Field(description="Input token symbol")
            token_out_symbol: str = Field(description="Output token symbol")
            fee_token_symbol: str = Field(description="Fee token symbol")
            amount_in: float = Field(description="Amount of the token to swap in")
            amount_out: float = Field(description="Amount of the token to swap out")
            fee_amount: float = Field(description="Fee amount")


class RoutingQuerier(FunctionWrapper[RoutingQueryArgs, RoutingResult]):
    """Query routing information from the jupiter service."""

    chain_config: ChainConfig
    base_url: str = "https://quote-api.jup.ag/v6"

    def __init__(self, *, chain_config: ChainConfig):
        self.chain_config = chain_config

        super().__init__()

    @classmethod
    def name(cls) -> LiteralString:
        return "get_swap_routing"

    @classmethod
    def description(cls) -> LiteralString:
        return "find the best routing for swapping tokens on Solana"

    @classmethod
    def notification(cls) -> str:
        return "\n*Query routing simulation on Jupiter aggregator...*\n"

    @staticmethod
    def _create_params(
        amount: float,
        swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]],
        token_in: TokenMetadata,
        token_out: TokenMetadata,
        slippage_bps: float = 0.5,
    ) -> Mapping:
        return {
            "amount": int(amount * 10 ** token_in.decimals),
            "swapMode": swap_mode,
            "slippageBps": int(slippage_bps * 100),
            "inputMint": token_in.address,
            "outputMint": token_out.address,
        }

    def _create_route_plan(self, route: dict) -> RoutingResult.Route:
        token_in_address: str = route["inputMint"]
        token_in = self.chain_config.get_token(None, token_in_address)
        if not token_in:
            raise ValueError(f"Input token not found: {token_in_address}")

        token_out_address: str = route["outputMint"]
        token_out = self.chain_config.get_token(None, token_out_address)
        if not token_out:
            raise ValueError(f"Output token not found: {token_out_address}")

        fee_token_address: str = route["feeMint"]
        fee_token = self.chain_config.get_token(None, fee_token_address)
        if not fee_token:
            raise ValueError(f"Fee token not found: {fee_token_address}")

        amount_in = float(route["inAmount"])
        amount_out = float(route["outAmount"])
        fee_amount = float(route["feeAmount"])
        return RoutingResult.Route(
            swap_info=RoutingResult.Route.SwapInfo(
                amm_address=route["ammKey"],
                label=route["label"],
                token_in_symbol=token_in.symbol,
                token_out_symbol=token_out.symbol,
                fee_token_symbol=fee_token.symbol,
                amount_in=float(amount_in) / 10 ** token_in.decimals,
                amount_out=float(amount_out) / 10 ** token_out.decimals,
                fee_amount=float(fee_amount) / 10 ** fee_token.decimals
            ),
            percent=route["percent"],
        )

    def _create_result(
        self,
        resp: Response,
        token_in: TokenMetadata,
        token_out: TokenMetadata,
    ) -> RoutingResult:
        if resp.status_code == 200:
            body: dict = resp.json()
            return RoutingResult(
                amount_in=float(body["inAmount"]) / 10 ** token_in.decimals,
                amount_out=float(body["outAmount"]) / 10 ** token_out.decimals,
                price_impact_pct=float(body["priceImpactPct"]),
                route_plan=[
                    self._create_route_plan(route)
                    for route in body["routePlan"]
                ],
            )
        else:
            raise RuntimeError(f"failed to query routing: status: {resp.status_code}, response: {resp.text}")

    @property
    def func(self) -> Optional[Callable[..., RoutingResult]]:
        def _query_routing(
            amount: float,
            swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]],
            token_in_symbol: str,
            token_out_symbol: str,
            slippage_bps: float = 0.5,
        ) -> RoutingResult:
            """Query routing information from the routing service."""
            token_in = self.chain_config.get_token(token_in_symbol, None, wrap=True)
            if not token_in:
                raise ValueError(f"Input token not found: {token_in_symbol}")
            token_out = self.chain_config.get_token(token_out_symbol, None, wrap=True)
            if not token_out:
                raise ValueError(f"Output token not found: {token_out_symbol}")
            resp = http_get(
                self.base_url + "/quote",
                params=self._create_params(
                    amount,
                    swap_mode,
                    token_in,
                    token_out,
                    slippage_bps,
                ),
            )
            return self._create_result(resp, token_in, token_out)

        return _query_routing

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[RoutingResult]]]:
        async def _query_routing(
            amount: float,
            swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]],
            token_in_symbol: str,
            token_out_symbol: str,
            slippage_bps: float = 0.5,
        ) -> RoutingResult:
            """Query routing information from the routing service."""
            async with AsyncClient() as client:
                token_in = self.chain_config.get_token(token_in_symbol, None)
                if not token_in:
                    raise ValueError(f"Input token not found: {token_in_symbol}")
                token_out = self.chain_config.get_token(token_out_symbol, None)
                if not token_out:
                    raise ValueError(f"Output token not found: {token_out_symbol}")
                resp = await client.get(
                    self.base_url + "/quote",
                    params=self._create_params(
                        amount,
                        swap_mode,
                        token_in,
                        token_out,
                        slippage_bps,
                    ),
                )
                return self._create_result(resp, token_in, token_out)

        return _query_routing
