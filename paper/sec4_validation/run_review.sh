#!/bin/bash
# The Monte Carlo tests (run_mc_review.py), two at a time, 300 realisations
# each (the default). The self-noise-like test is run_selfnoise.sh.
cd "$(dirname "$0")"
mkdir -p logs
printf '%s\n' \
  "offcov ar1 15" "offcov ar1 40" "offcov ar1 200" "offcov red 40" \
  "morph drift020 15" "morph drift020 40" "morph drift020 200" \
  | xargs -P 2 -I{} sh -c 'python3 run_mc_review.py {} > logs/review_$(echo {} | cut -d" " -f1-3 | tr " " _).log 2>&1'
