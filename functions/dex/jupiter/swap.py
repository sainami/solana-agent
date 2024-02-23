from httpx import get as http_get, post as http_post, AsyncClient
from typing import Any, Union, Literal, LiteralString, Mapping, Optional, Callable, Awaitable
from pydantic.v1 import BaseModel, Field, validator
from solders.pubkey import Pubkey
from langchain_core.callbacks.manager import CallbackManager, AsyncCallbackManager, handle_event, ahandle_event

from functions.wrapper import FunctionWrapper
from config.chain import ChainConfig, TokenMetadata


class SwapTxArgs(BaseModel):
    user_address: str = Field(description="Address or public-key of the user")
    amount: float = Field(description="Amount of the token to swap in or swap out")
    swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]] = Field(description="Type of the swap transaction")
    slippage_bps: float = Field(0.5, description="Slippage percentage tolerance")
    token_in_symbol: str = Field(description="Symbol of the token to swap in")
    token_out_symbol: str = Field(description="Symbol of the token to swap out")

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
    prompt: LiteralString = Field(description="Prompt message for the user to confirm the swap transaction")


class SwapTransaction(BaseModel):
    tx: str = Field(alias="swapTransaction")
    last_valid_height: int = Field(alias="lastValidBlockHeight")
    priority_fee_lamports: int = Field(alias="prioritizationFeeLamports")


class SwapTxBuilder(FunctionWrapper[SwapTxArgs, SwapTxResult]):
    """Build a swap transaction for a user to swap tokens on jupiter"""
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
        return "useful when you want to build a swap transaction on Jupiter for a user"

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
            slippage_bps: float = 0.5,
            token_in_symbol: Optional[str] = None,
            token_in_address: Optional[str] = None,
            token_out_symbol: Optional[str] = None,
            token_out_address: Optional[str] = None,
            callback_manager: Optional[CallbackManager] = None,
        ) -> SwapTxResult:
            """Build a swap transaction for a user to swap tokens on Jupiter"""
            token_in = self.chain_config.get_token(token_in_symbol, token_in_address)
            if not token_in:
                raise ValueError(f"Input token not found: {token_in_symbol} {token_in_address}")
            token_out = self.chain_config.get_token(token_out_symbol, token_out_address)
            if not token_out:
                raise ValueError(f"Output token not found: {token_out_symbol} {token_out_address}")
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
            if callback_manager:
                handle_event(
                    callback_manager.inheritable_handlers,
                    "send_metadata",
                    None,
                    SwapTransaction.parse_obj(resp.json()),
                )

            return SwapTxResult(prompt="Please approve the swap transaction")

        return _build_swap_tx

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[SwapTxResult]]]:
        async def _build_swap_tx(
            user_address: str,
            amount: float,
            swap_mode: Union[Literal["ExactIn"], Literal["ExactOut"]],
            slippage_bps: float = 0.5,
            token_in_symbol: Optional[str] = None,
            token_in_address: Optional[str] = None,
            token_out_symbol: Optional[str] = None,
            token_out_address: Optional[str] = None,
            callback_manager: Optional[AsyncCallbackManager] = None,
        ) -> SwapTxResult:
            """Build a swap transaction for a user to swap tokens on Jupiter"""
            async with AsyncClient() as client:
                token_in = self.chain_config.get_token(token_in_symbol, token_in_address)
                if not token_in:
                    raise ValueError(f"Input token not found: {token_in_symbol} {token_in_address}")
                token_out = self.chain_config.get_token(token_out_symbol, token_out_address)
                if not token_out:
                    raise ValueError(f"Output token not found: {token_out_symbol} {token_out_address}")

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
                if callback_manager:
                    await ahandle_event(
                        callback_manager.inheritable_handlers,
                        "send_metadata",
                        None,
                        SwapTransaction.parse_obj(resp.json()),
                    )

                return SwapTxResult(prompt="Please approve the swap transaction")

        return _build_swap_tx
