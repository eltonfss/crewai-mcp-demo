curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install build-essential -y
uv sync
uv tool install crewai
uv pip install crewai-tools[mcp]
