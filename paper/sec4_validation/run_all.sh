#!/bin/bash
# The Section 4 Monte Carlo runs: run_mc.py (k_c fixed), then run_mc_kc.py (k_c per realisation)
# and run_window.py (fit windows), for both bursts at every S/N. The two bursts run in parallel,
# one core each (about 1.5 h each). Logs go to logs/, results to results/.
cd "$(dirname "$0")"
mkdir -p results logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

chain() {
    tag=$1
    for snr in 15 25 40 70 120 200; do
        python3 run_mc.py "$tag" "$snr" > "logs/mc_${tag}_${snr}.txt" 2>&1
    done
    for snr in 15 25 40 70 120 200; do
        python3 run_mc_kc.py "$tag" "$snr" > "logs/kc_${tag}_${snr}.txt" 2>&1
    done
    for snr in 15 40 200; do
        python3 run_window.py "$tag" "$snr" > "logs/window_${tag}_${snr}.txt" 2>&1
    done
}

chain single &
chain double &
wait
echo done > logs/done.txt
