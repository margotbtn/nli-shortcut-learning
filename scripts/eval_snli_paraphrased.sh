#!/usr/bin/env bash
# Evaluate a model on the SNLI anti-shortcut paraphrased split.
#
# Shortcut tokens in hypotheses are replaced with neutral WordNet synonyms.
# Only fluent paraphrases (GPT-2 perplexity ratio < threshold) are retained.
# Requires: pip install nltk
#
# Usage: ./scripts/eval_snli_paraphrased.sh [input_view] [checkpoint] [run_dir]
#
# Examples:
#   ./scripts/eval_snli_paraphrased.sh pair best results/2026-01-16_233608_train_snli_bert-base-uncased
#   ./scripts/eval_snli_paraphrased.sh hypothesis_only best results/2026-01-25_171635_train_hypothesis_only_snli_bert-base-uncased
#
# Defaults: input_view=pair, checkpoint=best, run_dir required when checkpoint != pretrained
set -euo pipefail

INPUT_VIEW="${1:-pair}"
CHECKPOINT="${2:-best}"
RUN_DIR="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

case "${INPUT_VIEW}" in
  pair|hypothesis_only) ;;
  *)
    echo "Error: input_view must be 'pair' or 'hypothesis_only'."
    exit 1
    ;;
esac

case "${CHECKPOINT}" in
  pretrained|best|latest) ;;
  *)
    echo "Error: checkpoint must be 'pretrained', 'best', or 'latest'."
    exit 1
    ;;
esac

if [[ "${CHECKPOINT}" != "pretrained" && -z "${RUN_DIR}" ]]; then
  echo "Error: run_dir is required when checkpoint is not 'pretrained'."
  exit 1
fi

EVAL_ARGS=(
  --input_view "${INPUT_VIEW}"
  --checkpoint "${CHECKPOINT}"
  --mode "snli-paraphrased"
  --split "validation"
)

if [[ -n "${RUN_DIR}" ]]; then
  EVAL_ARGS+=(--training_run_dir "${RUN_DIR}")
fi

python "${REPO_ROOT}/src/training/eval.py" "${EVAL_ARGS[@]}"
