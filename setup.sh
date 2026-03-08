#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'

echo "==> repo root: ${ROOT_DIR}"

cd "${ROOT_DIR}"

echo "==> initializing submodules"
git submodule update --init --recursive

if command -v apt-get >/dev/null 2>&1; then
  echo "==> installing build tools"
  apt-get update
  apt-get install -y build-essential
fi

if [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ] && [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  echo "==> creating libcuda.so symlink"
  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
fi

if command -v claude >/dev/null 2>&1; then
  echo "==> claude already installed: $(command -v claude)"
else
  echo "==> installing claude code"
  curl -fsSL https://claude.ai/install.sh | bash
fi

if [ -f "${HOME}/.bashrc" ] && grep -Fqx "${PATH_EXPORT}" "${HOME}/.bashrc"; then
  echo "==> ~/.bashrc already exports ~/.local/bin"
else
  echo "==> adding ~/.local/bin to ~/.bashrc"
  printf '\n%s\n' "${PATH_EXPORT}" >> "${HOME}/.bashrc"
fi

echo "==> verifying claude installation"
claude --version

echo "==> installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "==> syncing autoresearch dependencies"
cd "${ROOT_DIR}/autoresearch"
uv sync

echo "==> preparing data and tokenizer"
uv run prepare.py

echo "==> verifying imports"
uv run python - <<'PY'
import importlib

modules = [
    "kernels",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "requests",
    "rustbpe",
    "tiktoken",
    "torch",
]

for name in modules:
    importlib.import_module(name)

print("imports ok")
PY

echo "==> setup complete"
echo "Claude Code is installed. Sign in manually by running:"
echo "  claude"
echo "The harness auto-detects the GPU count and uses that for both --gpus and --agents by default."
echo "Run the harness with:"
echo "  python3 -m harness.run --max-generations 1 --agent-timeout-minutes 20"
