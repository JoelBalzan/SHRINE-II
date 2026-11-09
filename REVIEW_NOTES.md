# Review notes

Items found during the readability refactor of 2026-11-09, and what was done about each.

## Fixed

| # | Item | Change |
|---|---|---|
| 1 | `profiles.crop` silently returned a short or empty crop when the peak was near either end of the voltages | Raises `ValueError` with the crop and the data's extent |
| 2 | `profiles.crop` left out the last time sample of `I` (SHRINE's `I[:, :-1]`). When `I` holds a whole number of ms, that also dropped the last 1-ms bin from the search | Sums all channels and all samples |
| 3 | `generate_profiles.py` did not require `-f` | `-f` is required, as in `maximise_structure.py` and `optimise_dm.py` |
| 4 | `-d/--DM` was a `float` in `maximise_structure.py` but a `str` elsewhere, and `-b/--bw` an `int` in two scripts but a `float` in the third | `-d` is a `str` and `-b` a `float` everywhere. `Initial DM:` now reads exactly as given (`0`, not `0.0`) |
| 6 | A negative `B[1] V B[1]` or `B[2] V B[2]` was clipped to 0 without a warning, giving sigma_DM = 0. That would also have made `check_grid` suggest a step of 0 | `dm_uncertainty` returns `ok = False` and says why |
| 7a | After a rise in the extended k_c search, if Mbar never fell back to the floor, `choose_kc` silently returned `N // 2` | Raises `RuntimeError`, like the S23 case; `-kc` still overrides it. This case is effectively unreachable |
| 10 | Profiles made by SHRINE's own `generate_profiles.py` (no `{label}_zap.txt`, channel 335 always dropped) were modelled with no zapped channels | `maximise_structure.py` warns, in the output and the summary file, when it has neither `--zap`/`--zap_edges` nor the zap file. The README says to use `--zap 335` |
| 11 | `optimise_dm.py` re-ran `choose_kc` on every pass and discarded the result, kept k_c by overwriting `args.force_kc`, and failed with `--max_passes 0` | k_c is chosen once on pass 1 (`maximise_structure.choose_kc_from_flags`) and passed to `estimate`. `--max_passes < 1` is rejected. Results are unchanged |
| 12 | `plots.plot_I_DM_t` drew each row half a trial-DM step off its DM | The image extends half a step beyond the first and last trial DMs |
| 13 | `checks/compare_final_sdm.py` crashed if every realisation of a case failed | It prints the failures instead |
| — | `profiles.US_PER_SAMPLE` was unused | Removed |

New tests: the crop finds a burst in the last millisecond, an out-of-range crop raises, and a non-positive variance is
reported as a failure (18 tests in all).

## Kept, and documented

| # | Item | Why kept, and where it is noted |
|---|---|---|
| 5 | The trial DMs are assumed uniformly spaced: V takes the step as `d[1] - d[0]`, unchecked | Every script makes uniform grids. Noted in the `dm_uncertainty` docstring and the README ("Uniform trial DMs") |
| 7b | sig_Mbar uses `Mbar[N//2 + 6:]` | Same as the paper's code (`kcvar/run_mc_kc_*.py`); the effect is negligible. Noted in `choose_kc` |
| 8 | `choose_kc` returns a 0-based index that `f(k)` uses as the 1-based mode number | S23's, SHRINE's and the paper's k_c values all follow this convention. Noted in `choose_kc` and the README's names table |
| 9 | `fine_freqs` labels each FFT bin half a bin above f_max − i·bw/n | Kept so that `--zap 335` still reproduces SHRINE's profiles exactly; the offset is ~5e-5 MHz for a 10-ms crop. Noted in `fine_freqs` |

## Other notes

- **Untracked tests:** `tests/` and `checks/` are listed in `.gitignore`, so they are not under version control.
  Remove those two lines to track them.
- **Bit-level results depend on how an expression is written.** NumPy may evaluate `a * b` with the operands
  swapped when one is a large temporary array (temporary elision). With fused multiply–add, complex products then
  differ in the last bit. The cause is inferred, not traced in NumPy's source. It was seen as ~1e-16 differences
  while checking the refactor bit for bit. For that reason one product in `checks/compare_final_sdm.py` keeps its
  temporary inline.
