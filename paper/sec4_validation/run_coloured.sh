#!/bin/bash
# The coloured-noise Monte Carlo (run_mc_coloured.py): both spectra at S/N = 15, 40 and 200, the
# two spectra in parallel, one core each. Logs go to logs/, results to results/.
cd "$(dirname "$0")"
mkdir -p results logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

chain() {
    spectrum=$1
    for snr in 15 40 200; do
        python3 run_mc_coloured.py "$spectrum" "$snr" > "logs/coloured_${spectrum}_${snr}.txt" 2>&1
    done
}

chain ar1 &
chain red &
wait
echo done > logs/coloured_done.txt
