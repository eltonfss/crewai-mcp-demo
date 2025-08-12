curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install build-essential -y
uv sync
uv tool install crewai
uv pip install crewai-tools[mcp]

# setup npx
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
nvm install node
nvm use node
npx --version
