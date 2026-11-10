# SHRINE-II

Structure maximisation of High-time Resolution Intensity profiles with No-nonsense Errors, revisited.

SHRINE ([Sutinjo et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...954...37S/abstract), "S23") finds the DM of an
FRB from its CRAFT voltages. SHRINE-II takes the same inputs and flags, and has SHRINE's two modes:

- **Structure maximisation**: the DM that maximises the structure of the profile, with S23's uncertainties replaced by
  the 1-sigma uncertainties of Balzan et al. (in prep.), "the paper" below:
  - sigma_S, the uncertainty of the structure parameter (paper Section 3.1);
  - sigma_DM, the uncertainty of the structure-maximising DM, from one parabola fit with the noise propagated through
    it (paper Section 3.3).
- **S/N maximisation**: the DM that maximises the boxcar S/N, as SHRINE's `maximise_sn.py` (with its bugs fixed). It
  runs on its own, and is the mode to use for very low S/N bursts, where structure maximisation does not work well.

The code is written to be read next to the paper: names follow the paper's symbols, and comments give the equation
or section each line evaluates.

## Install

```bash
pip install -e .            # or run the scripts in python/ directly from a checkout
python -m pytest            # optional: the tests in tests/
```
### Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
```

## Use

The scripts in `python/` take SHRINE's flags and file names, so they slot into SHRINE's pipeline.

```bash
# I(DM, t) from CRAFT X/Y voltages, for either mode (as SHRINE's generate_profiles.py)
python python/generate_profiles.py -l <label> -d <DM> -L -1 -H 1 --dDM 0.01 -t 25 -f <f0 MHz> -I <I file>

# Structure maximisation: the delta DM and its 1-sigma uncertainty (as SHRINE's maximise_structure.py)
python python/maximise_structure.py -l <label> -t 25 -d <DM> -f <f0 MHz> -s

# S/N maximisation, on its own (as SHRINE's maximise_sn.py)
python python/maximise_sn.py -l <label> -t 25 -s
```

Or in one go from the voltages with `optimise_dm.py`:

```bash
# Structure maximisation: generate_profiles.py and maximise_structure.py in one go
python python/optimise_dm.py -l <label> -d <DM> -L -1 -H 1 --dDM_auto -t 25 -f <f0 MHz> -I <I file> -s

# ... with the S/N-maximising DM too (as SHRINE's --do_sn)
python python/optimise_dm.py -l <label> -d <DM> -L -1 -H 1 --dDM_auto -t 25 -f <f0 MHz> -I <I file> --do_sn -s

# S/N maximisation only, on the trial DMs given (for very low S/N bursts)
python python/optimise_dm.py -l <label> -d <DM> -L -1 -H 1 --dDM 0.01 -t 25 -f <f0 MHz> -I <I file> --sn_only -s
```

| Script | Reads | Writes |
|---|---|---|
| `generate_profiles.py` | `{label}_X_t_{DM}.npy`, `{label}_Y_t_{DM}.npy`, the `-I` file (for the crop) | `{label}_DMs.npy/.dat`, `{label}_I_{dt}us.npy/.dat`, `{label}_zap.txt` (the zapped channels), `{label}_profile_summaryfile.txt` |
| `maximise_structure.py` | `{label}_DMs.npy`, `{label}_I_{dt}us.npy` | `{label}_structure_summaryfile.txt`, `{label}_SPs.dat`, `{label}_I_smooth_{dt}us.npy`; with `-s`, `DM.txt` and the plots below |
| `maximise_sn.py` | `{label}_DMs.npy`, `{label}_I_{dt}us.npy` | `{label}_SN_summaryfile.txt`, `{label}_SN_DMs.dat`, `{label}_SN.png`, `{label}_Max_SN_window.png`, `{label}_{dt}us_I_SN_max.png`; with `-s`, `{label}_I_smooth.npy` and `DM.txt` |
| `optimise_dm.py` | as `generate_profiles.py` | as `generate_profiles.py` (except its summary file) and `maximise_structure.py`; with `--do_sn`, as `maximise_sn.py` too; with `--sn_only`, as `generate_profiles.py` and `maximise_sn.py` only |

With `-s`, `maximise_structure.py` (and `optimise_dm.py`) write SHRINE's plots, `{label}_{dt}us_*.png`:

| Plot | Shows |
|---|---|
| `_structure_parameter.png` | S(d) with ±σ_S, the fit window, the parabola and d_fit ± σ_DM (as the paper's Figure 6) |
| `_DCT_v_k_index.png` | \|DCT coefficients\| of every profile against mode, with the k_c search (App. 1): the running mean M̄(k), the noise floor M_n and M_n + 4σ_M̄, S23's k_c, the extended search's k_c if it moved, and a `-kc` k_c; the k_c used is marked |
| `_I_max.png` | the profile and S x at the trial DM nearest d_fit |
| `_I_DM_t.png`, `_I_DM_t_smoothed.png` | I(DM, t) and S x over DM and t |
| `_detrended_noise.png` | the residual I − S x over DM and t (S23's residual; the paper argues it is not the noise) |
| `_noise_max.png` | the noise σ_t² is taken from: the profile at d_max reconstructed from its DCT modes k > k_min, rescaled to the full noise variance, with ±σ_t (paper Section 3.3.2) |

SHRINE's plots of S23's uncertainties (`_uncertainty`, `_relative_uncertainty`, `_relative_detrended_noise`,
`_adjusted_SP`) are not made, since those uncertainties are not computed, and nor is `_DM_index` (a straight line
for the uniformly spaced trial DMs every script makes).

Flags are SHRINE's, with these additions and changes:

- `--zap` and `--zap_edges`, in `generate_profiles.py`, `optimise_dm.py` and `maximise_structure.py`: coarse channels to
  leave out of the profiles. `--zap` takes indices and inclusive ranges (`--zap 0 100-110 335`); `--zap_edges N`
  leaves out the first and last N indices (`--zap_edges 1` removes channels 0 and 335). Channel c is centred on
  f_max − c MHz, so channels 1, 2, … run down from the top of the band and 335, 334, … up from the bottom, while
  channel 0 straddles both edges, half at each; so `--zap_edges N` trims N − ½ MHz from the top and N + ½ MHz from
  the bottom. Nothing is zapped by default; SHRINE
  always left out channel 335, so `--zap 335` gives SHRINE's profiles exactly. `generate_profiles.py` writes the
  zapped channels to `{label}_zap.txt`, and `maximise_structure.py` reads them from there unless given `--zap` or
  `--zap_edges` itself, since the channel delays in sigma_DM must describe the channels actually summed.

- `--dDM_auto`, in `generate_profiles.py` and `optimise_dm.py`: the delta-DM step becomes the
  one-sample DM step, the DM change that moves the bottom of the band by one output sample (`-t`) relative to the top,
  from `-f` and `-b` (`profiles.one_sample_dm_step`). It replaces `--dDM`/`--cDM`; `-L`/`-H` still set the range.
  For CRAFT's 336 MHz at 1271.5 MHz it is 0.00071 pc cm^-3 per us of `-t` (0.0071 at 10 us).
  Finer steps still change the profile, by fractions of a sample; when sigma_DM is smaller than this step, the
  summary suggests a finer `--dDM`.
- `--do_sn`, in `optimise_dm.py`: also find the S/N-maximising DM, as SHRINE's `--do_sn`.
- `--sn_only`, in `optimise_dm.py`: find only the S/N-maximising DM, with no structure maximisation (the standalone
  S/N mode).
- `-f/--f0` is required in `generate_profiles.py`, `maximise_structure.py` and `optimise_dm.py`: the profiles need it,
  and sigma_DM needs each channel's dispersion delay (with `-b/--bw` and `--nchan`).
- `-d/--DM` is read as text and `-b/--bw` as a number in every script, so the summary's `Initial DM:` is exactly what
  was given.
- `maximise_structure.py` warns if it has neither `--zap`/`--zap_edges` nor `{label}_zap.txt`. Profiles made by SHRINE's
  own `generate_profiles.py` have no zap file and always leave out channel 335, so give them `--zap 335`.
- `--s23_kc`, in `maximise_structure.py`, `maximise_sn.py` and `optimise_dm.py`: choose k_c with S23's procedure alone,
  without the paper's extended search (Appendix 1).

S23's own uncertainties are not computed.

The summary file keeps SHRINE's key lines (`kc:`, `Structure Maximising Delta DM:`,
`Uncertainty in Structure Maximising Delta DM: +/-...`, `Structure Parameter at Maximum: ... +/- ...`) and adds the fit
window, the relative curvature error, sqrt(<||e||^2>)/S and any warnings. If the trial DMs are too coarse, it suggests
`-L -H --dDM` for a second run of `generate_profiles.py` or `optimise_dm.py`. The trial DMs are never refined
automatically: the step is yours to set.

### S/N maximisation

`shrine2/snr.py` ports SHRINE's `maximise_sn.py`, and `python/maximise_sn.py` runs it on its own, on the output of
`generate_profiles.py` (or `optimise_dm.py --sn_only` from the voltages). It needs no structure-maximisation output.

- **When to use it.** For very low S/N bursts, where structure maximisation does not work well: the fit window or the
  parabola fails, or the summary's sqrt(<||e||^2>)/S is above ~0.3, so the expansion behind sigma_DM breaks down
  (paper Section 5.2). `maximise_structure.py` and `optimise_dm.py` then warn and point here.
- **Search.** For each trial DM, the smoothed profile S x, less a common baseline, is searched for the boxcar with the
  largest S/N = sum / (sigma sqrt(length)). Boxcars can start anywhere and run from 1 sample to 10 ms.
- **Result and range.** The S/N-maximising DM is the trial DM with the largest boxcar S/N. Its range is the trial DMs
  where the S/N stays within 1 of that maximum (SHRINE's criterion). The summary gives it as `+... -...`, and says if
  the S/N does not fall by 1 before the end of the trial DMs on a side (then that bound is a lower limit: widen
  `-L`/`-H`). Unlike sigma_DM, this range has not been calibrated against simulations. The step of the trial DMs
  should be well below the range.
- **k_c.** k_c for S x is chosen as for structure maximisation: `-kc`, or the automatic choice, with or without
  `--s23_kc`.
- **Kept from SHRINE.** sigma is the standard deviation of the residual I − S I over all trial DMs and samples, and the
  baseline is the smallest value of S I. With those choices it reproduces SHRINE's S/N exactly once the bugs below are
  put back in.
- **Fixed.**
  - SHRINE's boxcar of L samples summed L + 1 samples, divided by sqrt(L) and reported a length of L − 1. This
    inflated short-boxcar S/N by a few per cent and misreported the boxcar.
  - The range search could run past the ends of the trial DMs. It now stops there and says so.
  - The summary gave the boxcar's times in "us" when they were in ms.
  - The printed "S/N of the already denoised signal", which used the wrong slice, is dropped.

## How sigma_DM is computed

`shrine2.estimate.dm_uncertainty` runs the paper's Section 3.3 in order. Equation numbers are those of the draft of
2026-11-09; the LaTeX labels are given too, since the numbers will move.

`maximise_structure.py` and `optimise_dm.py` print each key result as it is found, labelled with its equation or
section: k_c (both steps of Appendix 1), the channel delays, Sp_max and d_max, sigma_t^2, sigma_S,
sqrt(<||e||^2>)/S, the fit window, the parabola, the correlation of the noise between trial DMs in V, std[c1],
sigma_DM and std[c2]/|c2|. In Python, pass `verbose=True` to `dm_uncertainty` and `choose_kc` for the same lines;
they are off by default so that Monte Carlo loops stay quiet.

| Step | Code | Paper |
|---|---|---|
| Sp(d_j) = \|\|D1 S x(d_j)\|\| at every trial DM | `structure.structure_parameter` | Eq. 4 (`eq:sp`), filter Eq. 3 (`eq:smooth`) |
| d_max, the trial DM with the largest Sp | `j_max = argmax(Sp)` | Sec. 2.2 |
| sigma_t^2 from the profile at d_max | `noise.noise_variance` | Sec. 3.3.2 (`sec:sigt`) |
| w at d_max, w = (D1 S x)^T D1 S / Sp | `structure.sp_gradient` | Eq. 28 (`eq:ahat`), Eq. 19 (`eq:weights`) |
| sigma_S = sqrt(sum_t w_t^2 sigma_t^2) | `noise.sigma_S` | Eq. 20 (`eq:sigS`) |
| fit window: contiguous trial DMs with Sp >= Sp_max - sigma_S | `fit.fit_window` | Sec. 3.3.3 (`sec:fit`) |
| parabola about its vertex: d_fit, c0, c1, c2, and B | `fit.fit_parabola` | Eqs. 29, 32 (`eq:parab`, `eq:lsq`) |
| w(t; d_j) at every trial DM in the window | `structure.sp_gradient` | Eq. 28 (`eq:ahat`) |
| V_jl = Cov[e_par(d_j), e_par(d_l)] | `covariance.noise_covariance` | App. 2, Eq. 41 (`eq:cov`) |
| std[c1]^2 = sum_jl B_1j B_1l V_jl | `B[1] @ V @ B[1]` | Eq. 34 (`eq:varc1`) |
| sigma_DM = std[c1] / \|2 c2\| | `sig_DM` | Eq. 35 (`eq:final`) |
| relative curvature error std[c2]/\|c2\| | `c2_err` | Eq. 36 (`eq:c2err`) |
| k_c: S23's procedure and the extended search | `structure.choose_kc` | App. 1, Eqs. 37, 38 (`eq:Mk`, `eq:runmean`) |

Names used throughout:

| Code | Paper | Meaning |
|---|---|---|
| `I` | x(d_j) | profiles, one row per trial DM |
| `d` | d_j | trial (delta-)DMs, uniformly spaced |
| `kc`, `f`, `lam` | k_c, f(k), λ_k | filter cutoff and response, eigenvalues of D1^T D1 (k_c is a 0-based index used as the mode number in f(k), as in S23, SHRINE and the paper) |
| `Sp`, `Sp_max` | 𝒮(d_j), 𝒮_max | structure parameter |
| `w`, `W` | w, w(t; d_j) | gradient of Sp with respect to each sample (the paper's coefficients), at one trial DM and at every trial DM in the window |
| `sig2` | σ_t² | profile-noise variance at each sample |
| `sig_S` | σ_𝒮 | structure parameter uncertainty |
| `win`, `n_win` | fit window | which trial DMs are fitted, and how many |
| `A`, `B`, `c` | A, B, c_0..c_2 | design matrix, least-squares matrix, fit parameters |
| `d_max`, `d_fit` | d_max, d_fit | largest-Sp trial DM, vertex of the fit |
| `dt_nu`, `share` | Δt_ν, — | channel delay per unit DM (samples), channel's share of the noise |
| `V` | V_jl | covariance between trial DMs |
| `sig_DM`, `c2_err` | σ_DM, std[c₂]/\|c₂\| | DM uncertainty, curvature check |

`profiles.py` is SHRINE's `generate_profiles.py`, written out step by step, and `profiles.channel_delays` gives the
delay and noise share of each channel it sums (channel 0 straddles the band edge and enters as two halves), leaving
out the zapped ones.

## Checks

- **Same numbers as the paper's code.** On the FIRES bursts and noise realisations behind the paper's Section 4
  (one component at S/N 15, 40 and 200, two components at S/N 40), `dm_uncertainty` returns the same d_fit, sigma_DM,
  sigma_S and fit window as the code used for the paper, to machine precision.
- **Same profiles as SHRINE.** `generate_profiles.py --zap 335` reproduces SHRINE's I(DM, t) from the same voltages,
  to single-precision rounding.
- **End to end on voltages.** For 160 noise realisations of a synthetic burst in CRAFT-like X/Y voltages (with self-noise),
  run through `generate_profiles` and Section 3.3 with SHRINE's channel model, the median quoted sigma_DM is
  1.01 ± 0.06 times the scatter of d_fit, and d_fit is unbiased. With the first and last 40 channels zapped
  (`--zap_edges 40`) it is 1.09 ± 0.07 and 1.07 ± 0.07 in two independent sets of 110 realisations; leaving the zapped
  channels in the channel model instead would raise sigma_DM by about 28%. The channel delays were checked against
  the shift that the dedispersion actually applies to each channel.
- **S/N maximisation.** On profiles from the synthetic voltages, `maximise_sn` picks the same trial DM and range as
  SHRINE's `maximise_sn.py`, and with SHRINE's boxcar off-by-one put back it gives the same maximum S/N to rounding
  (14 significant figures).
- The scripts for these are in `checks/`.
- `tests/`: Sp and w against the explicit matrices (w is the gradient of Sp), V against its definition and a
  Monte Carlo, the parabola fit, the k_c search, a toy-burst calibration, the boxcar search against brute force,
  the one-sample DM step and the crop.

## Using it on real data

- **Trial-DM step.** The paper's simulations use a step of sigma_DM/6, with trial DMs out to ~15 sigma_DM. You don't
  know sigma_DM before the first run; the scripts warn if the step is coarser than sigma_DM/3 and suggest a grid for a
  rerun. A coarser step still gives d_fit, with sigma_DM from fewer trial DMs in the fit window; at least three are
  needed.
- **Uniform trial DMs.** The trial DMs are assumed to be uniformly spaced, as every script makes them: the covariance V
  takes the step as d[1] − d[0], and this is not checked. If you make I(DM, t) yourself, keep the step constant.
- **The crop.** The burst is found as the brightest 1-ms bin of the `-I` file (assumed to be at 1 µs), summed over all
  channels, unless `--force_peak` is given. A crop that would run off either end of the voltages stops with an error.
- **Forcing k_c.** sigma_t^2 uses the DCT modes above k_min = max(N/4, 2 k_c), assumed to hold no burst power. A forced
  k_c well below the automatic one can break that and overestimate sigma_S and sigma_DM; a warning is written if the
  forced k_c is below the automatic one.
- **Channel noise.** V assumes each channel carries its share of the noise. `generate_profiles` normalises every channel
  to unit standard deviation over the crop, including the burst, so very bright channels are slightly down-weighted.
  Dead or RFI-flagged channels should be zapped (`--zap`): a channel of zeros would divide by zero there.
- **Self-noise.** sigma_t^2 is a local average from one profile, so with strong self-noise it varies from one trial DM
  to the next, and sigma_DM for a given burst can change by tens of per cent when the trial DMs change. In the voltage
  check above it was still unbiased on average.
- **Low S/N.** For very faint bursts use S/N maximisation (see above); the structure-maximisation scripts warn when
  sqrt(<||e||^2>)/S > 0.3 or std[c2]/|c2| > 0.5.
- **Validity.** sigma_DM is a first-order result. The summary reports sqrt(<||e||^2>)/S (paper Section 5.2; the
  simulations span ~0.07-0.3) and std[c2]/|c2| (unreliable as it approaches 1; Section 3.3.4). sigma_DM is the
  statistical uncertainty about the structure-maximising DM, which can differ from the dispersive DM when the burst
  changes across the band (Section 5.1).

## Credits

SHRINE: Timothy Perrott, Adrian Sutinjo, Danica Scott, Apurba Bera, Marcin Glowacki (Sutinjo et al. 2023). The profile
generation and k_c procedure here follow SHRINE's. Uncertainties: Balzan et al. (in prep.).
