#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
CONFIG="${2:-}"
CHECKPOINT="${3:-}"
RUN_DIR="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DEFAULT_CONFIG="${REPO_ROOT}/configs/base.yaml"

if [[ -z "${MODE}" || -z "${CONFIG}" ]]; then
  echo "Usage: $0 {train|eval} <config> [checkpoint] [run_dir]"
  echo "Defaults: checkpoint=pretrained, run_dir=none"
  exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="pretrained"
fi

if [[ "${MODE}" == "train" && "${CHECKPOINT}" == "best" ]]; then
  echo "Error: checkpoint 'best' is only valid for eval."
  exit 1
fi

if [[ "${CHECKPOINT}" != "pretrained" && -z "${RUN_DIR}" ]]; then
  echo "Error: run_dir is required when checkpoint is not 'pretrained'."
  exit 1
fi

case "${MODE}" in
  train)
    TRAIN_ARGS=(
      --config "${CONFIG}"
      --checkpoint "${CHECKPOINT}"
    )
    if [[ -n "${RUN_DIR}" ]]; then
      TRAIN_ARGS+=(--run_dir "${RUN_DIR}")
    fi
    python "${REPO_ROOT}/src/training/train.py" "${TRAIN_ARGS[@]}"
    ;;
  eval|evaluate)
    EVAL_ARGS=(
      --config "${CONFIG}"
      --checkpoint "${CHECKPOINT}"
    )
    if [[ -n "${RUN_DIR}" ]]; then
      EVAL_ARGS+=(--training_run_dir "${RUN_DIR}")
    fi
    python "${REPO_ROOT}/src/training/eval.py" "${EVAL_ARGS[@]}"
    ;;
  *)
    echo "Usage: $0 {train|eval} <config> [checkpoint] [run_dir]"
    exit 1
    ;;
esac
