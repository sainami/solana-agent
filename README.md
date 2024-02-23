# Solana Agent

## Installation
```bash
conda create -n solana-agent python=3.11
conda activate solana-agent
conda install -y 'langchain[all]' -c conda-forge
conda install -y pydantic -c conda-forge
pip install -U langchain-experimental
pip install -U langchain-community
pip install -U langchain-openai
pip install tavily-python
pip install httpx
pip install openai
pip install solana
pip install solders
pip install fastapi
pip install uvicorn
pip install argparse
pip install nest_asyncio
pip install mypy
pip install "redis[hiredis]"
```

## Export env
```bash
# win
conda env export --no-builds | findstr /v "prefix:" > environment.yml
# unix
conda env export --no-builds | grep -v "^prefix:" > environment.yml
```

## Import env
```bash
conda env create -f environment.yml
```

# TODO
- [ ] Refactor function modules to BaseModel