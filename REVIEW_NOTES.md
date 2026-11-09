# Review notes (readability refactor, 2026-11-09)

The refactor changed layout, names of local variables, comments and docstrings only. It did not change any
results: every public function, script output file (`.npy`, `.dat`, `.txt`, `.png`), printed line and `--help` text
was compared bit for bit before and after. The items below were noticed along the way and deliberately left
unchanged, for review.

## Possible bugs and questionable logic

1. **`profiles.crop` has no bounds check.** If the peak is within `crop_ms/2` of the start of the data,
   `mid - half` is negative and the slice silently returns a short or empty crop. A flat `-I` file does this:
   `argmax` gives 0, and the crop is empty. A peak near the end gives a short crop.
2. **`profiles.crop` drops the last time sample, not the last channel.** For `I` of shape `(n_chan, n_t)`,
   `I[:, :-1]` leaves out the last time sample. This is SHRINE's code. It may have been meant to drop the last
   channel, as `do_DM` does (speculative). It is harmless for finding the peak. The crop also assumes `I` is at
   1 µs (1000 samples per 1-ms bin).
3. **`generate_profiles.py` does not require `-f`.** `make_profiles` needs it, so a missing `-f` fails later
   with a `TypeError`. `maximise_structure.py` and `optimise_dm.py` do require it.
4. **Flag types differ between scripts.**
   - `-d/--DM` is a `str` in `generate_profiles.py` and `optimise_dm.py`, but a `float` in
     `maximise_structure.py`. So the summary's `Initial DM:` line reads `0` from one and `0.0` from the other.
   - `-b/--bw` is an `int` in two scripts and a `float` in `maximise_structure.py`.
5. **`dm_uncertainty` assumes uniform trial-DM spacing without checking.** It takes `step = d[1] - d[0]`, so a
   non-uniform grid silently gives a wrong V. It also needs at least two trial DMs.
6. **Negative variances are clipped silently.** A negative `B[1] V B[1]` or `B[2] V B[2]` becomes 0, giving
   sigma_DM = 0 or c2_err = 0 with no warning. This can only come from rounding or a V that is not positive
   semi-definite.
7. **Two `choose_kc` details.**
   - The `N // 2` fallback is effectively unreachable. It applies when Mbar never returns to the floor after the
     last rise, but Mbar over the upper half averages to about M_n. If it ever did apply, it would return
     `N // 2` silently.
   - sig_Mbar uses `Mbar[N//2 + 6:]`. That starts one window after the first six-mode window lying wholly in
     the upper half (`N//2 + 5`). The effect is negligible.
8. **`choose_kc` returns a 0-based index that `f(k)` uses as the 1-based mode number.** So k_c is one mode off
   the paper's numbering. This is inherited from SHRINE's `get_kc` and documented in the docstring.
9. **`fine_freqs` (SHRINE's `get_freqs`) labels each FFT bin half a bin high.** It gives
   f_max − (i − ½)·bw/n, where the convention used in `channel_delays` (DC at f_max) puts bin i at
   f_max − i·bw/n. The bin width is ~1e-4 MHz for a 10-ms crop, so this is negligible.
10. **Profiles from SHRINE's own `generate_profiles.py` have no `{label}_zap.txt`.** SHRINE always drops
    channel 335, so for such profiles `maximise_structure.py` needs `--zap 335`. Otherwise its channel model
    includes channel 335.
11. **`optimise_dm.py` passes after the first still run `choose_kc`.** k_c is fixed from pass 1 (by setting
    `args.force_kc`), so the result is discarded. If a later pass's automatic k_c is above pass 1's, no warning
    is shown. With `--max_passes 0` the loop never runs and the script fails with a `NameError`.
12. **`plots.plot_I_DM_t` draws each row half a trial-DM step off.** `extent` uses `d[0]` and `d[-1]` as the
    image edges, so each row is shifted by half a step from its trial DM. This is cosmetic.
13. **`checks/compare_final_sdm.py` fails if every realisation of a case fails.** The array of differences is
    then empty and `max()` raises.

## Technical debt

- **Unused constant:** `profiles.US_PER_SAMPLE` is never used.
- **No type annotations:** there are none anywhere, consistently; the docstrings give the shapes and units.
- **Untracked tests:** `tests/` and `checks/` are listed in `.gitignore`, so they are not under version control.
- **Bit-level results depend on how an expression is written.** NumPy may evaluate `a * b` with the operands
  swapped when one is a large temporary array (temporary elision). With fused multiply–add, complex products then
  differ in the last bit. This was seen while verifying: `X * np.exp(...)` and `X * ramps` differ by ~1e-16. The
  cause is inferred, not traced in NumPy's source. Naming an intermediate can therefore change results at the
  1e-16 level. The refactor was checked bit for bit for this reason, and in `checks/compare_final_sdm.py` one
  product is kept with the temporary inline so that its printed output is unchanged.
