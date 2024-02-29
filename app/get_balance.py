from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.rpc.responses import GetTokenAccountBalanceResp
from solders.token.associated import get_associated_token_address


async def main():
    client = AsyncClient("https://api.mainnet-beta.solana.com")
    account = Pubkey.from_string("FJBHhpLGrfbP2nb38Tc4ZejWPpj25T3ziruanYLQN2SJ")
    token_mint = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
    token_account = get_associated_token_address(account, token_mint)
    response = await client.get_token_account_balance(token_account)
    if isinstance(response, GetTokenAccountBalanceResp):
        print(response.value.ui_amount_string)
    else:
        print(0)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
