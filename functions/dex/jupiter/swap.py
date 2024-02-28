import logging

from httpx import get as http_get, post as http_post, AsyncClient, Response
from typing import Any, Union, List, Literal, LiteralString, Mapping, Optional, Callable, Awaitable
from pydantic.v1 import BaseModel, Field, validator
from solders.pubkey import Pubkey

from functions.wrapper import FunctionWrapper
from config.chain import ChainConfig, TokenMetadata


class SwapInfo(BaseModel):
    amm_address: str = Field(description="Address of the AMM pool")
    label: str = Field(description="Label of the AMM pool")
    token_in_symbol: str = Field(description="Input token symbol")
    token_out_symbol: str = Field(description="Output token symbol")
    fee_token_symbol: str = Field(description="Fee token symbol")
    amount_in: float = Field(description="Amount of the token to swap in")
    amount_out: float = Field(description="Amount of the token to swap out")
    fee_amount: float = Field(description="Fee amount")


class Route(BaseModel):
    swap_info: SwapInfo = Field(description="Swap information")
    percent: int = Field(description="Percentage of the swap")


class SwapRoute(BaseModel):
    swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]] = Field(description="Type of the swap event")
    amount_in: float = Field(description="Amount of the token to swap in")
    amount_out: float = Field(description="Amount of the token to swap out")
    price_impact_pct: float = Field(description="Price impact percentage")
    route_plan: List[Route] = Field(description="Swap route plan")


class SwapTxArgs(BaseModel):
    user_address: str = Field(description="Address or public-key of the user")
    amount: float = Field(description="Amount of the token to swap in or swap out")
    swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]] = Field(description="Type of the swap transaction")
    token_in_symbol: str = Field(description="Symbol of the token to swap in")
    token_out_symbol: str = Field(description="Symbol of the token to swap out")
    slippage_bps: float = Field(0.5, description="Slippage percentage tolerance")

    @validator("user_address", pre=True)
    @classmethod
    def check_user_address(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                pubkey = Pubkey.from_string(v)
                return str(pubkey)
            except Exception as e:
                raise ValueError(f"Invalid user address: {e}")
        else:
            raise TypeError(f"Invalid user address type {type(v)}")


class SwapTxResult(BaseModel):
    swap_route: SwapRoute = Field(description="Swap route simulation")
    swap_tx: str = Field(description="Swap transaction encoded in base64")
    last_valid_height: int = Field(description="Last valid block height")
    priority_fee: int = Field(description="Priority fee in lamports")


class SwapTxBuilder(FunctionWrapper[SwapTxArgs, SwapTxResult]):
    """Build a swap transaction for a user to swap tokens on jupiter"""
    return_direct: bool = True

    chain_config: ChainConfig
    base_url: str = "https://quote-api.jup.ag/v6"

    def __init__(self, *, chain_config: ChainConfig):
        self.chain_config = chain_config

        super().__init__()

    @classmethod
    def name(cls) -> LiteralString:
        return "swap_tx_builder"

    @classmethod
    def description(cls) -> LiteralString:
        return "create a swap transaction when user want to swap or buy tokens on Solana"

    @classmethod
    def notification(cls) -> str:
        return "\n*Preparing swap transaction on Jupiter, please confirm on you wallet...*\n"

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

    def _create_route_plan(self, route: dict) -> Route:
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
        return Route(
            swap_info=SwapInfo(
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

    def _create_swap_route(
        self,
        resp: Response,
        swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]],
        token_in: TokenMetadata,
        token_out: TokenMetadata,
    ) -> SwapRoute:
        if resp.status_code == 200:
            body: Mapping[str, Any] = resp.json()
            return SwapRoute(
                swap_mode=swap_mode,
                amount_in=float(body["inAmount"]) / 10 ** token_in.decimals,
                amount_out=float(body["outAmount"]) / 10 ** token_out.decimals,
                price_impact_pct=float(body["priceImpactPct"]),
                route_plan=[self._create_route_plan(route) for route in body["routePlan"]],
            )
        else:
            raise RuntimeError(f"failed to query routing: status: {resp.status_code}, response: {resp.text}")

    @property
    def func(self) -> Optional[Callable[..., SwapTxResult]]:
        def _build_swap_tx(
            user_address: str,
            amount: float,
            swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]],
            token_in_symbol: str,
            token_out_symbol: str,
            slippage_bps: float = 0.5,
        ) -> SwapTxResult:
            """Build a swap transaction for a user to swap tokens on Jupiter"""
            token_in = self.chain_config.get_token(token_in_symbol, None, wrap=True)
            token_out = self.chain_config.get_token(token_out_symbol, None, wrap=True)
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
            swap_route = self._create_swap_route(resp, swap_mode, token_in, token_out)

            resp = http_post(
                self.base_url + "/swap",
                json={
                    "userPublicKey": user_address,
                    "quoteResponse": resp.json(),
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"failed to query swap transaction: status: {resp.status_code}, response: {resp.text}"
                )
            data: Mapping[str, Any] = resp.json()
            return SwapTxResult(
                swap_route=swap_route,
                swap_tx=data["swapTransaction"],
                last_valid_height=data["lastValidBlockHeight"],
                priority_fee=data["prioritizationFeeLamports"],
            )

        return _build_swap_tx

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[SwapTxResult]]]:
        async def _build_swap_tx(
            user_address: str,
            amount: float,
            swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]],
            token_in_symbol: str,
            token_out_symbol: str,
            slippage_bps: float = 0.5,
        ) -> SwapTxResult:
            """Build a swap transaction for a user to swap tokens on Jupiter"""
            async with AsyncClient() as client:
                token_in = self.chain_config.get_token(token_in_symbol, None, wrap=True)
                token_out = self.chain_config.get_token(token_out_symbol, None, wrap=True)
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
                swap_route = self._create_swap_route(resp, swap_mode, token_in, token_out)

                resp = await client.post(
                    self.base_url + "/swap",
                    json={
                        "userPublicKey": user_address,
                        "quoteResponse": resp.json(),
                    },
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"failed to query swap transaction: status: {resp.status_code}, response: {resp.text}"
                    )
                data: Mapping[str, Any] = resp.json()
                res = SwapTxResult(
                    swap_route=swap_route,
                    swap_tx=data["swapTransaction"],
                    last_valid_height=data["lastValidBlockHeight"],
                    priority_fee=data["prioritizationFeeLamports"],
                )
                logging.info(f"Swap transaction created: {res.json(indent=2)}")
                return res

        return _build_swap_tx
