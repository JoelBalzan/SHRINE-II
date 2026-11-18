# Scripts for the paper

Every figure and number in Balzan et al. (in prep.), *Uncertainties of Structure-Maximising Dispersion Measures in Fast Radio Bursts*, comes from the scripts in this folder. They use `shrine2`, the code in this repository, for d_fit, σ_S and σ_DM: `shrine2.dm_uncertainty` for each noise realisation, and its building blocks (`fit_window`, `fit_parabola`, `coefficients`, `noise_covariance`) for the noiseless predictions and the fits over a given window.

## Running

The requirements are those of `shrine2` (numpy, scipy, matplotlib). `shrine2` is imported from this checkout, so it need not be installed. Each script can be run from any directory, and writes its figure to `figures/` and its numbers to `results/` in its own folder.

```bash
bash paper/run_all.sh                          # everything, about 4.5 h on two cores
python paper/sec4_validation/run_mc.py single 40 --n_real 20   # one quick case
```

Figure 3 and the Section 4 numbers need the Section 4 Monte Carlo (`sec4_validation/run_all.sh`) first. The other scripts are independent of it and take seconds to minutes. The result files of the paper's runs are included.

FIRES (https://github.com/JoelBalzan/FIRES) is needed only to regenerate the noiseless bursts (`bursts/simulate_bursts.sh`). The bursts themselves are in `bursts/`.

## Layout

| Folder | Contents |
|---|---|
| `sims.py` | Shared pieces: loading the bursts, channel delays, dedispersion, the noise level for a given S/N, the pilot (k_c, d_0, trial DMs), fits over a given window, and S23's uncertainties |
| `plotting.py` | The figure style (FRBop's) |
| `bursts/` | The noiseless FIRES bursts (`<tag>.npz`: `I`, `freq`, `time`), their FIRES configs, and `simulate_bursts.sh` |
| `sec2_s23/` | Section 2: Figures 1–5 |
| `sec3_uncertainties/` | Section 3: Figure 6 and the derivative-vs-slope numbers |
| `sec4_validation/` | Section 4: the Monte Carlo runs, Figures 7 and 11 (Appendix C), and the Section 4 numbers (also those of Appendix A, the abstract and the conclusions); also the coloured-noise Monte Carlo of Section 5.3 |
| `sec5_discussion/` | Section 5: d_0 against the true DM for scattering times 0–0.4 ms, the drifting sub-bursts (Figure 8) and the low-S/N numbers |
| `appA_kc/` | Appendix A: Figure 9 |
| `appB_cov/` | Appendix B: the Monte Carlo check of V (Figure 10 and its numbers) |

## Figures

| Figure | Label | File | Script |
|---|---|---|---|
| 1 | `fig:burst` | `fig_s23_burst.pdf` | `sec2_s23/make_burst_fig.py` |
| 2 | `fig:s23_vec` | `fig_s23_vec.pdf` | `sec2_s23/make_vec_fig.py` (schematic) |
| 3 | `fig:s23est` | `fig_s23_s23est.pdf` | `sec2_s23/make_s23est_fig.py` |
| 4 | `fig:s23_demo` | `fig_s23_demo.pdf` | `sec2_s23/make_demo_fig.py` |
| 5 | `fig:s23_response` | `fig_s23_response.pdf` | `sec2_s23/make_response_fig.py` (schematic) |
| 6 | `fig:fit` | `fig_s23_fit.pdf` | `sec3_uncertainties/make_fit_fig.py` |
| 7 | `fig:s23_val` | `fig_s23_validation.pdf` | `sec4_validation/make_validation_fig.py` |
| 8 | `fig:s23_drift` | `fig_s23_drift.pdf` | `sec5_discussion/make_drift_fig.py` |
| 9 | `fig:kc` | `fig_kc_search.pdf` | `appA_kc/make_kc_fig.py` |
| 10 | `fig:covmc` | `fig_cov_mc.pdf` | `appB_cov/make_cov_fig.py` (after `appB_cov/run_cov_mc.py white` and `selfnoise`) |
| 11 | `fig:s23_range` | `fig_s23_range.pdf` | `sec4_validation/make_range_fig.py` (Appendix C) |

## Numbers

| Where | What | Script | Result |
|---|---|---|---|
| §2.3 | S23's uncertainty of 𝒮 and their DM interval against the scatter of 𝒮 and of d_max (example burst) | `sec2_s23/make_s23est_fig.py` | `sec2_s23/results/fig_s23_s23est.json`: `ratio_Sp`, `ratio_dm` |
| §3.2 | The linear term alone moves the peak this fraction of the way (example burst) | `sec2_s23/make_demo_fig.py` | `sec2_s23/results/fig_s23_demo.json`: `peak_shift_ratio` |
| §3.3.4 | The derivative e∥′(d_0) overestimates σ_DM at S/N = 15 and 200 | `sec3_uncertainties/slope_numbers.py` | `sec3_uncertainties/results/slope_numbers.json`: `ratio` |
| §4.1, §5.1, abstract, §6 | Median quoted σ_S and σ_DM against the scatter; S23's factors (DM and 𝒮); the σ_S crossing range; spread of the quoted σ_DM; second-order term; fit-window size; coverage and bias; relative curvature error; d_0 against the true DM | `sec4_validation/summarise.py` | `sec4_validation/results/section4_numbers.json`: `paper` (ranges), `rows` (per case) |
| §4.1 | Narrower fit windows | `sec4_validation/summarise.py` (from `run_window.py`) | `section4_numbers.json`: `paper.narrower_windows_*`, `windows` |
| §4.2, App. A | k_c chosen per realisation, with and without the extended search | `sec4_validation/summarise.py` (from `run_mc_kc.py`) | `section4_numbers.json`: `paper.kc_*`, `rows[].kc_*` |
| §4.1 | The edges of S23's interval and of d_fit ± σ_DM shift the bottom of the band against the top by this much (example burst) | `sec4_validation/make_range_fig.py` | `sec4_validation/results/fig_s23_range.json`: `sweep_ms`, `sweep_samples_ours` |
| §4.1 | The burst's share of S23's residual (the script also gives the filtered noise power along â, ‖e‖ against σ_S and the size of the residual's noise part, no longer quoted) | `sec4_validation/s23_residual_numbers.py` | `sec4_validation/results/s23_residual_numbers.json` |
| §5.1 | d_fit against the true DM: coverage of d_true by d_fit ± kσ_DM (k = 1, 2, 3) and the offset in units of σ_MC, from the Section 4 runs | `sec5_discussion/dtrue_numbers.py` | `sec5_discussion/results/dtrue_numbers.json` |
| §5.1 | Noiseless d_0 − DM_true and σ_DM (pc cm⁻³ and in units of σ_DM) for both bursts at τ = 0, 0.05, 0.1, 0.2, 0.4 ms at 1 GHz, k_c from the pilot at each S/N; τ = 0 is the unscattered control | `sec5_discussion/tau_scan.py` (bursts from `bursts/simulate_bursts.sh`) | `sec5_discussion/results/tau_scan.json` |
| §5.2, §6 | d_0 of the drifting sub-bursts against the true DM, in pc cm⁻³ and in units of σ_DM | `sec5_discussion/run_drift.py` | `sec5_discussion/results/drift_results.json` |
| §4.5 | √⟨‖e‖²⟩/‖a‖ (the script also gives the fraction of the filtered noise power in modes where the burst's power exceeds the noise's, no longer quoted) | `sec5_discussion/lowsn_numbers.py` | `sec5_discussion/results/lowsn_numbers.json` |
| §4.4, §6 | Noise correlated between samples (AR(1), ρ = 0.5, and white plus red; one component at S/N 15, 40, 200): σ_S and σ_DM against the scatter with σ_t² from the profile, with the true variance per sample, and with the true Cov(n) (`dm_uncertainty(..., cov_n=...)`); the spread of σ_DM, the pull (d_fit − d_0)/σ_DM, coverage, √⟨‖e‖²⟩/‖a‖ and the second-order term | `sec4_validation/run_coloured.sh` (runs `run_mc_coloured.py`), then `sec4_validation/summarise_coloured.py` | `sec4_validation/results/coloured_numbers.json` |
| — | Not quoted in the paper since Appendix C was removed (the high-mode reconstruction near the profile ends; a fivefold variance bump; N = 601, k_min = N/4). Kept for reference | `sec3_uncertainties/dct_variance_numbers.py` | `sec3_uncertainties/results/dct_variance_numbers.json` |
| §4.1 (cited in §3.3.4) | OLS against GLS: vertex scatter and predicted σ for GLS with pseudo-inverses of V (one component, S/N 40, 200 realisations) | `sec3_uncertainties/gls_check.py` | `sec3_uncertainties/results/gls_check.json` |
| §4.1 | Pull, coverage of ±1σ and ±2σ for the white-noise runs | `sec4_validation/summarise.py` results (`mc_*.npz`) | computed from `d_fit`, `sig_DM` and `meta.d0` |
| §4.3, §6 | Noise whose variance follows the burst (5× at the peak, channel shares changing with t), with σ_t² estimated and true: σ_DM and σ_S against the scatter, coverage, pull, the predicted σ_DM against white noise, and the estimate of σ_t² against the true one | `sec4_validation/run_selfnoise.sh` (runs `run_mc_selfnoise.py`), then `sec4_validation/summarise_selfnoise.py` | `sec4_validation/results/selfnoise_numbers.json` |
| §5.2, §6 | The drifting burst (t_sep = 0.2 ms) at S/N 15, 40, 200 | `sec4_validation/run_review.sh` (runs `run_mc_review.py morph`), then `summarise_review.py` | `sec4_validation/results/review_numbers.json`: `morph_*` |
| §5.2 | Drifting burst at S/N 200: σ_DM with w from the noiseless burst | `sec4_validation/morph_ahat_check.py` | `sec4_validation/results/morph_ahat_check.json` |
| §4.4, §6 | Coloured noise with Cov(n) estimated from four off-pulse profiles (circulant, from the mean periodogram) | `sec4_validation/run_review.sh` (`run_mc_review.py offcov`), then `summarise_review.py` | `review_numbers.json`: `offcov_*` |
| App. A | The extended k_c search with its rise level, running-mean length and reach changed | `appA_kc/kc_sensitivity.py` | `appA_kc/results/kc_sensitivity.json` |
| App. B.1, §3.3.4, §4.3 | V against noise-only simulations (one component, S/N 40, 41 trial DMs over the noiseless fit window, 10,000 realisations): white noise in equal channels at even trial DMs, and variance 5× at the peak in unequal channels at uneven trial DMs. Correlations with d_ref, the change of std[e_par] with trial DM relative to d_ref, std[c1] and std[c2], and how much std[c1] depends on the change of std[e_par] | `appB_cov/run_cov_mc.py {white,selfnoise}` | `appB_cov/results/cov_mc_numbers.json` |

## How the simulations are set up

- **Bursts.** Each burst is a FIRES dynamic spectrum at its true DM (337 channels of 1 MHz from 1087.5 to 751.5 MHz, 10-µs samples, 6 ms), so a trial DM d is an offset from the true DM.
- **Dedispersion.** Dedispersing at d moves channel ν earlier by d Δt_ν samples, with Δt_ν measured from 919.5 MHz, by a Fourier phase ramp (so fractional shifts are exact), and sums the channels (`sims.dedisperse`, `sims.dedisperse_grid`).
- **Noise.** White Gaussian noise of equal variance in every channel, at the level FIRES uses for a given S/N: the boxcar holding 95% of the fluence in the frequency-summed profile has that S/N (`sims.channel_noise_sigma`).
- **The pilot** (`sims.pilot`), for each burst and S/N:
  - k_c is the median of S23's k_c (`shrine2.choose_kc` without the extended search) over five noise realisations on coarse trial DMs, then held fixed.
  - d_0 and the predicted σ_DM come from Section 3.3 applied to the noiseless burst with the true σ_t² (`sims.noiseless_fit`): the trial DMs within σ_S of the peak of ‖a(d)‖, the vertex of the parabola fitted over them, and σ_DM from `noise_covariance`.
  - The trial DMs are spaced by σ_DM/6 out to max(15 σ_DM, 20 one-sample steps) either side of d_0, with d_0 on the grid; they are refined until σ_DM changes by less than 1%.
- **Seeds.** Each case of Section 4 has its own seed (`sims.CASES`). The pilot draws five realisations from it and the Monte Carlo continues from there, so `run_mc.py`, `run_mc_kc.py` and `run_window.py` see the same noise realisations. The example burst (Figures 1, 3, 4, 6 and 8) is realisation `sims.EXAMPLE_SEED` = 111 of the one-component burst at S/N = 40.
- **S23.** Their uncertainty of 𝒮 and their DM interval are in `sims.s23_relative_uncertainty` and `sims.s23_interval`, as in SHRINE.
- **Noise whose variance follows the burst (§4.3).** `run_mc_selfnoise.py` gives each channel variance σ_ch²[1 + g I_ν(t)²] at the true DM, with one g per burst such that the frequency-summed profile's variance peaks at 5× off-pulse (at every S/N), and dedisperses the noise with the burst. σ_DM is evaluated with σ_t² from the profile and with the true σ_t² of the profile at each realisation's d_max (the exact variance after the fractional channel shifts: each channel's variance convolved with the square of its shift filter). k_c is that of the white-noise pilot; d_0 and the trial DMs come from the noiseless fit with the true σ_t². The noise seeds are those of `sims.CASES` plus 2000.
- **Report tests.** `run_mc_review.py` has two tests, each with its own seed (600 or 700 plus the S/N): `offcov` (the coloured noise of `run_mc_coloured.py` with Cov(n) estimated from four off-pulse profiles per realisation) and `morph` (white noise on the drifting burst `drift020`). k_c comes from the white-noise pilot; d_0 and the trial DMs from the noiseless fit with the true noise.
- **Coloured noise (§4.4).** `run_mc_coloured.py` filters unit white noise in each channel by the square root of the noise's power spectrum, circularly, so that the circular dedispersion leaves its covariance unchanged and the true Cov(n) is circulant. k_c and d_0 are those of the white-noise pilot; the trial DMs are spaced by σ_DM/6 of the noiseless burst with the true Cov(n). The noise seeds are those of `sims.CASES` plus 1000.
