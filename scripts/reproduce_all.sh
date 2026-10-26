#!/usr/bin/env bash
# Explicit stages only. No arguments prints help and performs no experiments.
set -euo pipefail
cd "$(dirname "$0")/.."

hw2_python="${HW2_PYTHON:-python}"
hw2_data="${HW2_DATA:-data/prepared}"
hw2_output="${HW2_OUTPUT:-outputs}"
hw2_basic="${HW2_BASIC_CHECKPOINT:-runs/basic-final}"
hw2_bonus="${HW2_BONUS_CHECKPOINT:-runs/bonus-final}"

hw2_command() { "$hw2_python" -m deepmir_hw2 "$@"; }

case "${1:-help}" in
  prepare)
    hw2_command prepare --dataset POP909/training --output "$hw2_data"
    ;;
  train-basic|train-bonus)
    : "${CUDA_VISIBLE_DEVICES:?Set the authorized physical GPU index explicitly}"
    task="${1#train-}"
    hw2_command train --config "configs/$task.json" --data "$hw2_data" --output "runs/$task"
    ;;
  final-basic|final-bonus)
    : "${CUDA_VISIBLE_DEVICES:?Set the authorized physical GPU index explicitly}"
    : "${HW2_FINAL_STEPS:?Set the final-stage step count after assessing convergence}"
    task="${1#final-}"
    hw2_command train --config "configs/$task.json" --data "$hw2_data" \
      --init-from "runs/$task" --all-training --steps "$HW2_FINAL_STEPS" \
      --output "runs/$task-final"
    ;;
  generate)
    while read -r setting top_k temperature; do
      hw2_command generate --checkpoint "$hw2_basic" --mode unconditional --count 20 --bars 32 \
        --top-k "$top_k" --temperature "$temperature" --seed 2026 \
        --output "$hw2_output/basic-a/$setting"
      hw2_command generate --checkpoint "$hw2_basic" --mode continue \
        --prompts POP909/prompts/basic-b --bars 24 --top-k "$top_k" \
        --temperature "$temperature" --seed 2026 --output "$hw2_output/basic-b/$setting"
    done <<'SETTINGS'
A 20 0.9
B 50 0.9
C 50 1.1
SETTINGS
    hw2_command generate --checkpoint "$hw2_bonus" --mode accompany \
      --prompts POP909/prompts/bonus --top-k 50 --temperature 0.9 --seed 2026 \
      --output "$hw2_output/bonus"
    ;;
  evaluate)
    hw2_command reference --data "$hw2_data" --count 100 --bars 32 --output "$hw2_output/real"
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
      '        generate, evaluate, render A|B|C' \
      'Commands are a reproduction plan, not evidence that experiments have run.'
    ;;
  *) printf 'Unknown stage: %s\n' "$1" >&2; exit 2 ;;
esac
