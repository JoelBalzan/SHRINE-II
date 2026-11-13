# Scripts for the paper

Every figure and number in Balzan et al. (in prep.), *Uncertainties of Structure-Maximising Dispersion Measures in Fast Radio Bursts*, comes from the scripts in this folder. They use `shrine2`, the code in this repository, for d_fit, σ_S and σ_DM: `shrine2.dm_uncertainty` for each noise realisation, and its building blocks (`fit_window`, `fit_parabola`, `coefficients`, `noise_covariance`) for the noiseless predictions and the fits over a given window.

## Running

The requirements are those of `shrine2` (numpy, scipy, matplotlib). `shrine2` is imported from this checkout, so it need not be installed. Each script can be run from any directory, and writes its figure to `figures/` and its numbers to `results/` in its own folder.

```bash
bash paper/run_all.sh                          # everything, about 1.5 h on two cores
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
| `sec4_validation/` | Section 4: the Monte Carlo runs, Figures 7 and 8, and the Section 4 numbers (also those of Appendix A, the abstract and the conclusions) |
| `sec5_discussion/` | Section 5: the drifting sub-bursts (Figure 9) and the low-S/N numbers |
| `appA_kc/` | Appendix A: Figure 10 |

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
| 8 | `fig:s23_range` | `fig_s23_range.pdf` | `sec4_validation/make_range_fig.py` |
| 9 | `fig:s23_drift` | `fig_s23_drift.pdf` | `sec5_discussion/make_drift_fig.py` |
| 10 | `fig:kc` | `fig_kc_search.pdf` | `appA_kc/make_kc_fig.py` |

## Numbers

| Where | What | Script | Result |
|---|---|---|---|
| §2.3 | S23's uncertainty of 𝒮 and their DM interval against the scatter of 𝒮 and of d_max (example burst) | `sec2_s23/make_s23est_fig.py` | `sec2_s23/results/fig_s23_s23est.json`: `ratio_Sp`, `ratio_dm` |
| §3.2 | The linear term alone moves the peak this fraction of the way (example burst) | `sec2_s23/make_demo_fig.py` | `sec2_s23/results/fig_s23_demo.json`: `peak_shift_ratio` |
| §3.3.4 | The derivative e∥′(d_0) overestimates σ_DM at S/N = 15 and 200 | `sec3_uncertainties/slope_numbers.py` | `sec3_uncertainties/results/slope_numbers.json`: `ratio` |
| §4, abstract, §6 | Median quoted σ_S and σ_DM against the scatter; S23's factors (DM and 𝒮); the σ_S crossing range; spread of the quoted σ_DM; second-order term; fit-window size; coverage and bias; relative curvature error; d_0 against the true DM | `sec4_validation/summarise.py` | `sec4_validation/results/section4_numbers.json`: `paper` (ranges), `rows` (per case) |
| §4 | Narrower fit windows | `sec4_validation/summarise.py` (from `run_window.py`) | `section4_numbers.json`: `paper.narrower_windows_*`, `windows` |
| §4, App. A | k_c chosen per realisation, with and without the extended search | `sec4_validation/summarise.py` (from `run_mc_kc.py`) | `section4_numbers.json`: `paper.kc_*`, `rows[].kc_*` |
| §4 | The edges of S23's interval and of d_fit ± σ_DM shift the bottom of the band against the top by this much (example burst) | `sec4_validation/make_range_fig.py` | `sec4_validation/results/fig_s23_range.json`: `sweep_ms`, `sweep_samples_ours` |
| §4 | Filtered noise power along â; ‖e‖ against σ_S; the burst's share of S23's residual and the size of its noise part | `sec4_validation/s23_residual_numbers.py` | `sec4_validation/results/s23_residual_numbers.json` |
| §5.1, §6 | d_0 of the drifting sub-bursts against the true DM, in pc cm⁻³ and in units of σ_DM | `sec5_discussion/run_drift.py` | `sec5_discussion/results/drift_results.json` |
| §5.2 | √⟨‖e‖²⟩/‖a‖, and the fraction of the filtered noise power in modes where the burst's power exceeds the noise's | `sec5_discussion/lowsn_numbers.py` | `sec5_discussion/results/lowsn_numbers.json` |

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
