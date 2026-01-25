#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
INPUT_VIEW="${2:-}"
CHECKPOINT="${3:-}"
RUN_DIR="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 [mode] [input_view] [checkpoint] [run_dir]"
  echo "Defaults: input_view=pair, checkpoint=pretrained, run_dir=none"
  exit 1
fi

case "${INPUT_VIEW}" in
  pair|hypothesis_only)
    ;;
  *)
    echo "Error: input_view must be 'pair' or 'hypothesis_only'."
    exit 1
    ;;
esac

case "${CHECKPOINT}" in
  pretrained|best|latest)
    ;;
  *)
    echo "Error: checkpoint must be 'pretrained', 'best' or 'latest'."
    exit 1
    ;;
esac

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
      --input_view "${INPUT_VIEW}"
      --checkpoint "${CHECKPOINT}"
    )
    if [[ -n "${RUN_DIR}" ]]; then
      TRAIN_ARGS+=(--run_dir "${RUN_DIR}")
    fi
    python "${REPO_ROOT}/src/training/train.py" "${TRAIN_ARGS[@]}"
    ;;
  eval)
    EVAL_ARGS=(
      --input_view "${INPUT_VIEW}"
      --checkpoint "${CHECKPOINT}"
    )
    if [[ -n "${RUN_DIR}" ]]; then
      EVAL_ARGS+=(--training_run_dir "${RUN_DIR}")
    fi
    python "${REPO_ROOT}/src/training/eval.py" "${EVAL_ARGS[@]}"
    ;;
  *)
    echo "Usage: $0 {train|eval} [input_view] [checkpoint] [run_dir]"
    exit 1
    ;;
esac
