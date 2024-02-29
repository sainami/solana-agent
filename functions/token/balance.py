from typing import Any, List, Mapping, Optional, LiteralString, Callable, Awaitable
from pydantic.v1 import BaseModel, Field, validator
from solders.pubkey import Pubkey
from solders.rpc.responses import GetBalanceResp, GetTokenAccountBalanceResp
from solders.token.associated import get_associated_token_address
from solana.rpc.api import Client
from solana.rpc.async_api import AsyncClient

from config.chain import ChainConfig
from functions.wrapper import FunctionWrapper


class BalanceArgs(BaseModel):
    account: str = Field(description="The account address to query balance for")
    token_symbols: List[str] = Field(description="The symbol list of the tokens pending query")

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


class BalanceResult(BaseModel):
    balances: Mapping[str, str] = Field(description="The token balances of the account")


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
        return "query_token_balances"

    @classmethod
    def description(cls) -> LiteralString:
        return "query token balances based on symbols on Solana"

    @classmethod
    def notification(cls) -> str:
        return "\n*Querying token balances in batch...*\n"

    @property
    def func(self) -> Optional[Callable[..., BalanceResult]]:
        if self.client:
            def _get_balance(
                account: str,
                token_symbols: List[str],
            ) -> BalanceResult:
                assert self.client is not None

                account = Pubkey.from_string(account)
                balances = {}
                for symbol in token_symbols:
                    token = self.chain_config.get_token(symbol, None)
                    if token:
                        token_mint = Pubkey.from_string(token.address)
                        token_account = get_associated_token_address(account, token_mint)
                        resp = self.client.get_token_account_balance(token_account)
                        if isinstance(resp, GetTokenAccountBalanceResp):
                            balances[symbol] = resp.value.ui_amount_string
                        else:
                            balances[symbol] = "0"
                    else:
                        resp = self.client.get_balance(account)
                        if isinstance(resp, GetBalanceResp):
                            balances[symbol] = str(resp.value / 10 ** self.chain_config.chain.coin_decimals)
                        else:
                            balances[symbol] = "0"
                return BalanceResult(balances=balances)

            return _get_balance

    @property
    def async_func(self) -> Optional[Callable[..., Awaitable[BalanceResult]]]:
        if self.async_client:
            async def _get_balance(
                account: str,
                token_symbols: List[str],
            ) -> BalanceResult:
                assert self.async_client is not None

                account = Pubkey.from_string(account)
                balances = {}
                for symbol in token_symbols:
                    token = self.chain_config.get_token(symbol, None)
                    if token:
                        token_mint = Pubkey.from_string(token.address)
                        token_account = get_associated_token_address(account, token_mint)
                        resp = await self.async_client.get_token_account_balance(token_account)
                        if isinstance(resp, GetTokenAccountBalanceResp):
                            balances[symbol] = resp.value.ui_amount_string
                        else:
                            balances[symbol] = "0"
                    else:
                        resp = await self.async_client.get_balance(account)
                        if isinstance(resp, GetBalanceResp):
                            balances[symbol] = str(resp.value / 10 ** self.chain_config.chain.coin_decimals)
                        else:
                            balances[symbol] = "0"
                return BalanceResult(balances=balances)

            return _get_balance
