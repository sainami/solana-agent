from typing import Any, Optional, LiteralString, Callable, Awaitable
from pydantic.v1 import BaseModel, Field, validator
from solders.pubkey import Pubkey
from solders.token.associated import get_associated_token_address
from solana.rpc.api import Client
from solana.rpc.async_api import AsyncClient

from config.chain import ChainConfig
from functions.wrapper import FunctionWrapper


class BalanceArgs(BaseModel):
    account: str = Field(description="The account address to query balance for")
    token_symbol: Optional[str] = Field(None, description="The symbol of the token")
    token_address: Optional[str] = Field(None, description="The address of the token")

    @validator("account", pre=True)
    @classmethod
    def check_account(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                pubkey = Pubkey.from_string(v)
                return str(pubkey)
            except Exception as e:
                raise ValueError(f"Invalid account address: {e}")
        else:
            raise TypeError(f"Invalid account address type {type(v)}")

    @validator("token_address", pre=True)
    @classmethod
    def check_token_contract(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                pubkey = Pubkey.from_string(v)
                return str(pubkey)
            except Exception as e:
                raise ValueError(f"Invalid token address: {e}")
        elif v is not None:
            raise TypeError(f"Invalid token address type {type(v)}")


class BalanceResult(BaseModel):
    balance: str = Field(description="The balance of the account")


class BalanceGetter(FunctionWrapper[BalanceArgs, BalanceResult]):
    chain_config: ChainConfig
    client: Optional[Client]
    async_client: Optional[AsyncClient]

    def __init__(
        self,
        *,
        chain_config: ChainConfig,
        client: Optional[Client] = None,
        async_client: Optional[AsyncClient] = None,
    ):
        if client is None and async_client is None:
            raise ValueError("either client or async_client must be provided")
        self.chain_config = chain_config
        self.client = client
        self.async_client = async_client

        super().__init__()

    @classmethod
    def name(cls) -> LiteralString:
        return "get_token_balance"

    @classmethod
    def description(cls) -> LiteralString:
        return "useful for when you query some token balance on Solana"

    @classmethod
    def notification(cls) -> str:
        return "\n*Querying token balance...*\n"

    @property
    def func(self) -> Optional[Callable[..., BalanceResult]]:
        if self.client:
            def _get_balance(
                account: str,
                token_symbol: Optional[str] = None,
                token_address: Optional[str] = None,
            ) -> BalanceResult:
                assert self.client is not None

                account = Pubkey.from_string(account)
                token = self.chain_config.get_token(token_symbol, token_address)
                if token:
                    # SPL token balance
                    token_mint = Pubkey.from_string(token.address)
                    token_account = get_associated_token_address(account, token_mint)
                    resp = self.client.get_token_account_balance(token_account)
                    balance = resp.value.ui_amount_string
                else:
                    # native coin balance
                    resp = self.client.get_balance(account)
                    balance = str(resp.value / 10 ** self.chain_config.chain.coin_decimals)
                return BalanceResult(balance=balance)

            return _get_balance

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[BalanceResult]]]:
        if self.async_client:
            async def _get_balance(
                account: str,
                token_symbol: Optional[str] = None,
                token_address: Optional[str] = None,
            ) -> BalanceResult:
                assert self.async_client is not None

                account = Pubkey.from_string(account)
                token = self.chain_config.get_token(token_symbol, token_address)
                if token:
                    # SPL token balance
                    token_mint = Pubkey.from_string(token.address)
                    token_account = get_associated_token_address(account, token_mint)
                    resp = await self.async_client.get_token_account_balance(token_account)
                    balance = resp.value.ui_amount_string
                else:
                    # native coin balance
                    resp = await self.async_client.get_balance(account)
                    balance = str(resp.value / 10 ** self.chain_config.chain.coin_decimals)
                return BalanceResult(balance=balance)

            return _get_balance
