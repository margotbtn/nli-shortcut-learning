#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_baseline.sh train
#   ./scripts/run_baseline.sh eval

MODE="${1:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${MODE}" in
  train)
    python "${REPO_ROOT}/src/training/train.py"
    ;;
  eval|evaluate)
    python "${REPO_ROOT}/src/training/evaluate.py"
    ;;
  *)
    echo "Usage: $0 {train|eval}"
    exit 1
    ;;
esac
