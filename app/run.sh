#!/bin/zsh

export OPENAI_API_VERSION="2024-02-15-preview"
export AZURE_OPENAI_API_KEY="1af48d1cbce14f46a03d6e65bee707a5"
export AZURE_OPENAI_ENDPOINT="https://hc-instanceeastus.openai.azure.com"

export GOOGLE_API_KEY="AIzaSyCYX5aXzm4P0CHt7G1-G1y7M0EGGrIjMoA"
export GOOGLE_CSE_ID="13234ba1a46e648d0"

export TAVILY_API_KEY="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"

conda run --no-capture-output -n solana-agent \
python app/run_svc.py \
  --log-level=INFO \
  --debug-mode=true \
  --host="0.0.0.0" \
  --port=8105 \
  --model-config=".config/model.json" \
  --chain-config=".config/chain.json" \
  --rpc="https://api.mainnet-beta.solana.com"
