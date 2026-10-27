#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

"${HW2_PYTHON:-python}" -m deepmir_hw2 audit "$@"
