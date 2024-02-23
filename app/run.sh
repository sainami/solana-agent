#!/bin/zsh

export OPENAI_API_KEY="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"
export TAVILY_API_KEY="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"

conda run --no-capture-output -n solana-agent \
python app/run_svc.py \
  --log-level=INFO \
  --debug-mode=true \
  --host="0.0.0.0" \
  --port=8105 \
  --model-config=".config/model.json" \
  --chain-config=".config/chain.json" \
  --rpc="https://rpc.ankr.com/solana/3d3768859f11aef1e3cdf0a25e9f5d5691ec8357424bd5fd9447fc6c26114461"
