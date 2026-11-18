#!/bin/bash
# The Monte Carlo with noise whose variance follows the burst (run_mc_selfnoise.py, Section 4.2):
# the one-component burst at S/N = 15, 40 and 200 and the two-component burst at S/N = 40, in two
# chains, one core each. Logs go to logs/, results to results/.
cd "$(dirname "$0")"
mkdir -p results logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run() {
    python3 run_mc_selfnoise.py "$1" "$2" > "logs/selfnoise_$1_$2.txt" 2>&1
}

(run single 15; run single 200) &
(run single 40; run double 40) &
wait
echo done > logs/selfnoise_done.txt
