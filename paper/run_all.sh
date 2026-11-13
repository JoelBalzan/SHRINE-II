#!/bin/bash
# Every figure and number of the paper, in order: the Section 4 Monte Carlo first (about 1.5 h on two
# cores), because Figure 3 and the Section 4 numbers read its results; then the rest (a few minutes).
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

sec4_validation/run_all.sh
python3 sec4_validation/summarise.py
python3 sec4_validation/make_validation_fig.py
python3 sec4_validation/make_range_fig.py
python3 sec4_validation/s23_residual_numbers.py

python3 sec2_s23/make_burst_fig.py
python3 sec2_s23/make_vec_fig.py
python3 sec2_s23/make_s23est_fig.py
python3 sec2_s23/make_demo_fig.py
python3 sec2_s23/make_response_fig.py

python3 sec3_uncertainties/make_fit_fig.py
python3 sec3_uncertainties/slope_numbers.py

python3 sec5_discussion/run_drift.py
python3 sec5_discussion/make_drift_fig.py
python3 sec5_discussion/lowsn_numbers.py

python3 appA_kc/make_kc_fig.py
