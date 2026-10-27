#!/usr/bin/env bash
# Explicit stages only. No arguments prints help and performs no experiments.
set -euo pipefail
cd "$(dirname "$0")/.."

hw2_python="${HW2_PYTHON:-python}"
hw2_data="${HW2_DATA:-data/prepared}"
hw2_output="${HW2_OUTPUT:-outputs}"
hw2_basic="${HW2_BASIC_CHECKPOINT:-runs/basic-final}"
hw2_bonus="${HW2_BONUS_CHECKPOINT:-runs/bonus-final}"
hw2_basic_model="${HW2_BASIC_MODEL:-models/basic.pt}"
hw2_bonus_model="${HW2_BONUS_MODEL:-models/bonus.pt}"
hw2_numeric_strategy="${HW2_NUMERIC_STRATEGY:-sample}"

hw2_command() { "$hw2_python" -m deepmir_hw2 "$@"; }

resolve_source_commit() {
  if [[ -n "${HW2_SOURCE_COMMIT:-}" ]]; then
    export HW2_SOURCE_COMMIT
    return
  fi
  if ! HW2_SOURCE_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null)"; then
    printf '%s\n' 'Set HW2_SOURCE_COMMIT when training outside a Git checkout.' >&2
    exit 2
  fi
  export HW2_SOURCE_COMMIT
}

require_final() {
  "$hw2_python" -m deepmir_hw2.experiment_stage require-final --checkpoint "$1"
}

latest_checkpoint_file() {
  local run_dir="$1"
  local checkpoint_name
  checkpoint_name="$("$hw2_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
    "$run_dir/latest.json")"
  printf '%s/%s\n' "$run_dir" "$checkpoint_name"
}

export_basic() {
  local final_checkpoint_file
  final_checkpoint_file="$(latest_checkpoint_file "$hw2_basic")"
  require_final "$final_checkpoint_file"
  hw2_command export --checkpoint "$final_checkpoint_file" --output "$hw2_basic_model"
}

export_bonus() {
  local final_checkpoint_file
  final_checkpoint_file="$(latest_checkpoint_file "$hw2_bonus")"
  require_final "$final_checkpoint_file"
  hw2_command export --checkpoint "$final_checkpoint_file" --output "$hw2_bonus_model"
}

validate_numeric_options() {
  case "$hw2_numeric_strategy" in
    sample|beam) ;;
    *) printf 'Unknown numeric strategy: %s\n' "$hw2_numeric_strategy" >&2; exit 2 ;;
  esac
}

generate_basic() {
  validate_numeric_options
  while read -r setting top_k temperature; do
    hw2_command generate --checkpoint "$hw2_basic_model" --mode unconditional --count 20 --bars 32 \
      --top-k "$top_k" --temperature "$temperature" --numeric-strategy "$hw2_numeric_strategy" \
      --seed 2026 --output "$hw2_output/basic-a/$setting"
    hw2_command generate --checkpoint "$hw2_basic_model" --mode continue \
      --prompts POP909/prompts/basic-b --bars 24 --top-k "$top_k" \
      --temperature "$temperature" --numeric-strategy "$hw2_numeric_strategy" \
      --seed 2026 --output "$hw2_output/basic-b/$setting"
  done <<SETTINGS
A ${HW2_BASIC_A_TOP_K:-20} ${HW2_BASIC_A_TEMPERATURE:-0.9}
B ${HW2_BASIC_B_TOP_K:-50} ${HW2_BASIC_B_TEMPERATURE:-0.9}
C ${HW2_BASIC_C_TOP_K:-50} ${HW2_BASIC_C_TEMPERATURE:-1.1}
SETTINGS
}

generate_bonus() {
  validate_numeric_options
  hw2_command generate --checkpoint "$hw2_bonus_model" --mode accompany \
    --prompts POP909/prompts/bonus --top-k "${HW2_BONUS_TOP_K:-50}" \
    --temperature "${HW2_BONUS_TEMPERATURE:-0.9}" \
    --numeric-strategy "$hw2_numeric_strategy" \
    --seed 2026 --output "$hw2_output/bonus"
}

case "${1:-help}" in
  prepare)
    hw2_command prepare --dataset POP909/training --output "$hw2_data"
    ;;
  train-basic|train-bonus)
    : "${CUDA_VISIBLE_DEVICES:?Set the authorized physical GPU index explicitly}"
    resolve_source_commit
    task="${1#train-}"
    hw2_command train --config "configs/$task.json" --data "$hw2_data" --output "runs/$task"
    ;;
  final-basic|final-bonus)
    : "${CUDA_VISIBLE_DEVICES:?Set the authorized physical GPU index explicitly}"
    resolve_source_commit
    task="${1#final-}"
    if [[ "$task" == basic ]]; then
      final_checkpoint="$hw2_basic"
    else
      final_checkpoint="$hw2_bonus"
    fi
    hw2_command train --config "configs/$task-full-corpus.json" --data "$hw2_data" \
      --output "$final_checkpoint"
    ;;
  export-basic)
    export_basic
    ;;
  export-bonus)
    export_bonus
    ;;
  generate-basic)
    generate_basic
    ;;
  generate-bonus)
    generate_bonus
    ;;
  generate)
    generate_basic
    generate_bonus
    ;;
  evaluate)
    hw2_command reference --data "$hw2_data" --output "$hw2_output/real"
    hw2_command evaluate --task basic --input "$hw2_output/real" \
      --output "$hw2_output/metrics/real.csv"
    for setting in A B C; do
      hw2_command evaluate --task basic --input "$hw2_output/basic-a/$setting" \
        --output "$hw2_output/metrics/$setting.csv"
    done
    hw2_command evaluate --task bonus --input "$hw2_output/bonus" --prompts POP909/prompts/bonus \
      --output "$hw2_output/metrics/bonus.csv"
    ;;
  render)
    : "${HW2_SOUNDFONT:?Set an explicit SoundFont path}"
    setting="${2:?Specify the selected basic-A setting: A, B or C}"
    case "$setting" in A|B|C) ;; *) exit 2 ;; esac
    hw2_command render --input "$hw2_output/basic-a/$setting" \
      --output "$hw2_output/basic-a/$setting" --soundfont "$HW2_SOUNDFONT"
    for setting in A B C; do
      hw2_command render --input "$hw2_output/basic-b/$setting" \
        --output "$hw2_output/basic-b/$setting" --soundfont "$HW2_SOUNDFONT"
    done
    hw2_command render --input "$hw2_output/bonus" --output "$hw2_output/bonus" \
      --soundfont "$HW2_SOUNDFONT"
    ;;
  help)
    printf '%s\n' 'Stages: prepare, train-basic, train-bonus, final-basic, final-bonus,' \
      '        export-basic, export-bonus, generate-basic, generate-bonus, generate,' \
      '        evaluate, render A|B|C' \
      'Commands are a reproduction plan, not evidence that experiments have run.'
    ;;
  *) printf 'Unknown stage: %s\n' "$1" >&2; exit 2 ;;
esac
