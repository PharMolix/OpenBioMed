#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash local_tools/run_tool.sh <tool.py> [args...]" >&2
  exit 2
fi

TOOL_NAME="$1"
shift
TOOL_PATH="$SCRIPT_DIR/$TOOL_NAME"

if [ ! -f "$TOOL_PATH" ]; then
  echo "Tool not found: $TOOL_PATH" >&2
  exit 2
fi

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "No usable Python interpreter found. Tried: python3, python" >&2
  exit 127
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 126
fi

exec "$PYTHON_BIN" "$TOOL_PATH" "$@"
