from httpx import get as http_get, post as http_post, AsyncClient
from typing import Any, Union, Literal, LiteralString, Mapping, Optional, Callable, Awaitable
from pydantic.v1 import BaseModel, Field, validator
from solders.pubkey import Pubkey

from functions.wrapper import FunctionWrapper
from config.chain import ChainConfig, TokenMetadata


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
    raw_tx: str = Field(description="Swap transaction encoded in base64")
    last_valid_height: int = Field(description="Last valid block height")
    priority_fee: int = Field(description="Priority fee in lamports")


# class SwapTransaction(BaseModel):
#     tx: str = Field(alias="swapTransaction")
#     last_valid_height: int = Field(alias="lastValidBlockHeight")
#     priority_fee_lamports: int = Field(alias="prioritizationFeeLamports")


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
        return "\n*Building swap transaction...*\n"

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
            if resp.status_code != 200:
                raise RuntimeError(
                    f"failed to query swap routing: status: {resp.status_code}, response: {resp.text}"
                )

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
            data: dict = resp.json()
            return SwapTxResult(
                raw_tx=data["swapTransaction"],
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
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"failed to query swap routing: status: {resp.status_code}, response: {resp.text}"
                    )

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
                data: dict = resp.json()
                return SwapTxResult(
                    raw_tx=data["swapTransaction"],
                    last_valid_height=data["lastValidBlockHeight"],
                    priority_fee=data["prioritizationFeeLamports"],
                )

        return _build_swap_tx
