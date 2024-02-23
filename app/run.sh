#!/bin/zsh

export OPENAI_API_KEY="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"
export TAVILY_API_KEY="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"

conda run --no-capture-output -n web3-agent \
python app/run_svc.py \
  --debug-mode=true \
  --host="0.0.0.0" \
  --port=8105 \
  --model-config=".config/model.json" \
  --chain-config=".config/chain.json"
