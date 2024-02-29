FROM condaforge/mambaforge

LABEL authors="Lone"

WORKDIR /solana-agent

COPY . /solana-agent

RUN mamba env create -f /solana-agent/environment.yml

EXPOSE 8901

ENV OPENAI_API_KEY="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" \
    TAVILY_API_KEY="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"

ENTRYPOINT [ \
    "conda", "run", "--no-capture-output", "-n", "solana-agent", \
    "python", "/solana-agent/app/run_svc.py", "--log-level=INFO", "--host=127.0.0.1", "--port=8901", \
    "--model-config=/solana-agent/.config/model.json", "--chain-config=/solana-agent/.config/chain.json",  \
    "--rpc=https://api.mainnet-beta.solana.com" \
]