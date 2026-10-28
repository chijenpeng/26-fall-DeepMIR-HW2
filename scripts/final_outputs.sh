#!/usr/bin/env bash
# Submission outputs from models/*.pt.
# Usage: bash scripts/final_outputs.sh <basic-a|basic-b|bonus|no-chord|render|metrics> [OUT]
# WAVs: `render` (needs FluidSynth; SF=/path/to/font.sf2) converts every MIDI under OUT.
set -euo pipefail
part=$1
out=${2:-outputs/final}
gen="python -m deepmir_hw2 generate"

settings() {
  case $1 in
    A) echo "--top-k 20 --temperature 0.9" ;;
    B) echo "--top-p 0.9 --temperature 1.0" ;;
    C) echo "--top-k 50 --temperature 1.2" ;;
  esac
}

case $part in
  basic-a)
    for s in A B C; do
      $gen --checkpoint models/basic.pt --mode unconditional --count 20 --bars 32 \
        $(settings $s) --output "$out/basic-a/$s"
    done ;;
  basic-b)
    for s in A B C; do
      $gen --checkpoint models/basic.pt --mode continue --prompts POP909/prompts/basic-b \
        --bars 24 $(settings $s) --output "$out/basic-b/$s"
    done ;;
  bonus)
    for s in A B C; do
      $gen --checkpoint models/bonus.pt --mode accompany --prompts POP909/prompts/bonus \
        $(settings $s) --output "$out/bonus/$s"
    done ;;
  no-chord)
    $gen --checkpoint models/basic-no-chord.pt --mode unconditional --count 20 --bars 32 \
      $(settings A) --output "$out/basic-a-no-chord/A" ;;
  render)
    find "$out" -name '*.mid' | while read -r midi; do
      fluidsynth -ni -F "${midi%.mid}.wav" -T wav -r 44100 "$SF" "$midi" > /dev/null
    done ;;
  metrics)
    for d in "$out"/basic-a/* "$out"/basic-a-no-chord/*; do
      python -m deepmir_hw2 evaluate basic --input "$d" --output "$d/metrics.csv"
    done
    for d in "$out"/bonus/*; do
      python -m deepmir_hw2 evaluate bonus --input "$d" --output "$d/metrics.csv"
    done ;;
  *) echo "unknown part: $part" >&2; exit 1 ;;
esac
