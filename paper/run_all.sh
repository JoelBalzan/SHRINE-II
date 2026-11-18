#!/bin/bash
# Every figure and number of the paper, in order: the Section 4 Monte Carlo first (about 1.5 h on two
# cores), because Figure 3 and the Section 4 numbers read its results; then the rest (a few minutes);
# then the coloured-noise Monte Carlo of Section 5.3 (about 1 h on two cores); then the Monte Carlo with
# noise whose variance follows the burst (Section 4.2; about 35 min on two cores); then the tests added for
# the report (off-pulse Cov(n), the drifting burst, OLS vs GLS, the k_c settings; about 1.5 h on two
# cores).
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

python3 appB_cov/run_cov_mc.py white
python3 appB_cov/run_cov_mc.py selfnoise
python3 appB_cov/make_cov_fig.py

bash sec4_validation/run_coloured.sh
python3 sec4_validation/summarise_coloured.py

bash sec4_validation/run_selfnoise.sh
python3 sec4_validation/summarise_selfnoise.py

bash sec4_validation/run_review.sh
python3 sec4_validation/summarise_review.py
python3 sec4_validation/morph_ahat_check.py
python3 sec3_uncertainties/dct_variance_numbers.py
python3 sec3_uncertainties/gls_check.py
python3 appA_kc/kc_sensitivity.py
