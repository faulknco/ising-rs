#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

cd "$ROOT"

[ -d "$VENV" ] || uv venv "$VENV"

uv pip compile requirements.txt -o requirements.lock.txt
uv pip sync --python "$VENV/bin/python" requirements.lock.txt
uv pip install --python "$VENV/bin/python" pip-audit
"$VENV/bin/pip-audit" -r requirements.lock.txt --no-deps --disable-pip
"$VENV/bin/pip-audit"
