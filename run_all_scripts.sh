#!/bin/bash

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

for script in scripts/0{2,3,4,5,6,7,8,9}_*.py scripts/1[0-9]_*.py; do
  if [ -f "$script" ]; then
    echo "Running $script..."
    PYTHONPATH="$ROOT" python "$script" || exit 1
  fi
done

echo "Complete."
