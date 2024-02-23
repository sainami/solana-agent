import json
import httpx


if __name__ == "__main__":
    resp = httpx.get("https://token.jup.ag/strict")
    with open("token.json", "w") as f:
        token_list = []
        for token in resp.json():
            token_list.append({
                "name": token["name"],
                "symbol": token["symbol"],
                "address": token["address"],
                "decimals": token["decimals"],
            })
        f.write(json.dumps(token_list, indent=2))
