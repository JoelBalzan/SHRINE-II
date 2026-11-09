# SHRINE-II

Structure maximisation of High-time Resolution Intensity profiles with No-nonsense Errors, revisited.

SHRINE ([Sutinjo et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...954...37S/abstract), "S23") finds the DM that
maximises the structure of an FRB's profile. SHRINE-II takes the same inputs and flags, and replaces S23's uncertainties
with the 1-sigma uncertainties of Balzan et al. (in prep.), "the paper" below:

- sigma_S, the uncertainty of the structure parameter (paper Section 3.1);
- sigma_DM, the uncertainty of the structure-maximising DM, from one parabola fit with the noise propagated through it
  (paper Section 3.3).

The code is written to be read next to the paper: names follow the paper's symbols, and comments give the equation
or section each line evaluates.

## Install

```bash
pip install -e .            # or run the scripts in python/ directly from a checkout
python -m pytest            # optional: the tests in tests/
```

Needs numpy, scipy and matplotlib.

## Use

The scripts in `python/` take SHRINE's flags and file names, so they slot into SHRINE's pipeline.

```bash
# 1. I(DM, t) from CRAFT X/Y voltages (as SHRINE's generate_profiles.py)
python python/generate_profiles.py -l <label> -d <DM> -L -1 -H 1 --dDM 0.01 -t 25 -f <f0 MHz> -I <I file>

# 2. Structure-maximising delta DM and its 1-sigma uncertainty (as SHRINE's maximise_structure.py)
python python/maximise_structure.py -l <label> -t 25 -d <DM> -f <f0 MHz> -s

# or both in one go, refining the trial DMs until they are fine enough, with the S/N-maximising DM too
python python/optimise_dm.py -l <label> -d <DM> -L -1 -H 1 --dDM_auto -t 25 -f <f0 MHz> -I <I file> --do_sn -s

# S/N-maximising delta DM and the comparison plot (as SHRINE's maximise_sn.py and structure_SN_comparison.py)
python python/maximise_sn.py -l <label> -t 25 -s
python python/structure_SN_comparison.py -l <label> -t 25 -i_st <row> -i_sn <row> -d <label>_DMs.npy -s <label>_I_smooth_25us.npy
```

| Script | Reads | Writes |
|---|---|---|
| `generate_profiles.py` | `{label}_X_t_{DM}.npy`, `{label}_Y_t_{DM}.npy`, the `-I` file (for the crop) | `{label}_DMs.npy/.dat`, `{label}_I_{dt}us.npy/.dat`, `{label}_zap.txt` (the zapped channels), `{label}_profile_summaryfile.txt` |
| `maximise_structure.py` | `{label}_DMs.npy`, `{label}_I_{dt}us.npy` | `{label}_structure_summaryfile.txt`, `{label}_SPs.dat`, `{label}_I_smooth_{dt}us.npy`; with `-s`, `DM.txt` and plots |
| `maximise_sn.py` | `{label}_DMs.npy`, `{label}_I_{dt}us.npy` | `{label}_SN_summaryfile.txt`, `{label}_SN_DMs.dat`, `{label}_SN.png`, `{label}_Max_SN_window.png`, `{label}_{dt}us_I_SN_max.png`; with `-s`, `{label}_I_smooth.npy` and `DM.txt` |
| `structure_SN_comparison.py` | the trial DMs and smoothed profiles given | `{label}_structure_SN_comparison.png` |
| `optimise_dm.py` | as `generate_profiles.py` | as `generate_profiles.py` and `maximise_structure.py`, for the last pass; earlier passes' profiles with `_pass<n>`; with `--do_sn`, as `maximise_sn.py` (on pass 1) and the comparison plot |

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
  Finer steps still change the profile, by fractions of a sample, and when sigma_DM is smaller than this step,
  `optimise_dm.py` refines below it.
- `--do_sn`, in `optimise_dm.py`: also find the S/N-maximising DM, as SHRINE's `--do_sn`.
- `-f/--f0` is required in `maximise_structure.py`: sigma_DM needs each channel's dispersion delay (with `-b/--bw` and `--nchan`).
- `--s23_kc`, in `maximise_structure.py`, `maximise_sn.py` and `optimise_dm.py`: choose k_c with S23's procedure alone,
  without the paper's extended search (Appendix 1).

S23's own uncertainties are not computed.

The summary file keeps SHRINE's key lines (`kc:`, `Structure Maximising Delta DM:`,
`Uncertainty in Structure Maximising Delta DM: +/-...`, `Structure Parameter at Maximum: ... +/- ...`) and adds the fit
window, the relative curvature error, sqrt(<||e||^2>)/S and any warnings. If the trial DMs are too coarse, it suggests
`-L -H --dDM` for a second run of `generate_profiles.py`, which `optimise_dm.py` does automatically.

### S/N maximisation

`shrine2/snr.py` ports SHRINE's `maximise_sn.py`. It is not part of the paper's method and is kept for comparison.

- **Search.** For each trial DM, the smoothed profile S x, less a common baseline, is searched for the boxcar with the
  largest S/N = sum / (sigma sqrt(length)). Boxcars can start anywhere and run from 1 sample to 10 ms.
- **Result and range.** The S/N-maximising DM is the trial DM with the largest boxcar S/N. Its range is the trial DMs
  where the S/N stays within 1 of that maximum.
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

| Step | Code | Paper |
|---|---|---|
| Sp(d_j) = \|\|D1 S x(d_j)\|\| at every trial DM | `structure.structure_parameter` | Eq. 4 (`eq:sp`), filter Eq. 3 (`eq:smooth`) |
| d_max, the trial DM with the largest Sp | `j_max = argmax(Sp)` | Sec. 2.2 |
| sigma_t^2 from the profile at d_max | `noise.noise_variance` | Sec. 3.3.2 (`sec:sigt`) |
| w at d_max, w = (D1 S x)^T D1 S / Sp | `structure.coefficients` | Eq. 28 (`eq:ahat`), Eq. 19 (`eq:weights`) |
| sigma_S = sqrt(sum_t w_t^2 sigma_t^2) | `noise.sigma_S` | Eq. 20 (`eq:sigS`) |
| fit window: contiguous trial DMs with Sp >= Sp_max - sigma_S | `fit.fit_window` | Sec. 3.3.3 (`sec:fit`) |
| parabola about its vertex: d_fit, c0, c1, c2, and B | `fit.parabola` | Eqs. 29, 32 (`eq:parab`, `eq:lsq`) |
| w(t; d_j) at every trial DM in the window | `structure.coefficients` | Eq. 28 (`eq:ahat`) |
| V_jl = Cov[e_par(d_j), e_par(d_l)] | `covariance.covariance` | App. 2, Eq. 41 (`eq:cov`) |
| std[c1]^2 = sum_jl B_1j B_1l V_jl | `B[1] @ V @ B[1]` | Eq. 34 (`eq:varc1`) |
| sigma_DM = std[c1] / \|2 c2\| | `sig_DM` | Eq. 35 (`eq:final`) |
| relative curvature error std[c2]/\|c2\| | `c2_err` | Eq. 36 (`eq:c2err`) |
| k_c: S23's procedure and the extended search | `structure.choose_kc` | App. 1, Eqs. 37, 38 (`eq:Mk`, `eq:runmean`) |

Names used throughout:

| Code | Paper | Meaning |
|---|---|---|
| `I` | x(d_j) | profiles, one row per trial DM |
| `d` | d_j | trial (delta-)DMs, uniformly spaced |
| `kc`, `f`, `lam` | k_c, f(k), λ_k | filter cutoff and response, eigenvalues of D1^T D1 |
| `Sp`, `Sp_max` | 𝒮(d_j), 𝒮_max | structure parameter |
| `w`, `W` | w, w(t; d_j) | coefficients at one trial DM, at every trial DM in the window |
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
  Monte Carlo, the parabola fit, the k_c search, a toy-burst calibration, the boxcar search against brute force, and
  the one-sample DM step.

## Using it on real data

- **Trial-DM step.** The paper's simulations use a step of sigma_DM/6, with trial DMs out to ~15 sigma_DM. You don't
  know sigma_DM before the first run, so either rerun `generate_profiles.py` with the suggested grid or use
  `optimise_dm.py`. It warns if the step is coarser than sigma_DM/3.
- **Forcing k_c.** sigma_t^2 uses the DCT modes above k_min = max(N/4, 2 k_c), assumed to hold no burst power. A forced
  k_c well below the automatic one can break that and overestimate sigma_S and sigma_DM; a warning is written if the
  forced k_c is below the automatic one.
- **Channel noise.** V assumes each channel carries its share of the noise. `generate_profiles` normalises every channel
  to unit standard deviation over the crop, including the burst, so very bright channels are slightly down-weighted.
  Dead or RFI-flagged channels should be zapped (`--zap`): a channel of zeros would divide by zero there.
- **Self-noise.** sigma_t^2 is a local average from one profile, so with strong self-noise it varies from one trial DM
  to the next, and sigma_DM for a given burst can change by tens of per cent when the trial DMs change. In the voltage
  check above it was still unbiased on average.
- **Validity.** sigma_DM is a first-order result. The summary reports sqrt(<||e||^2>)/S (paper Section 5.2; the
  simulations span ~0.07-0.3) and std[c2]/|c2| (unreliable as it approaches 1; Section 3.3.4). sigma_DM is the
  statistical uncertainty about the structure-maximising DM, which can differ from the dispersive DM when the burst
  changes across the band (Section 5.1).

## Credits

SHRINE: Timothy Perrott, Adrian Sutinjo, Danica Scott, Apurba Bera, Marcin Glowacki (Sutinjo et al. 2023). The profile
generation and k_c procedure here follow SHRINE's. Uncertainties: Balzan et al. (in prep.).
