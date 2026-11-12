# Checks

One-off checks behind the "Checks" section of the main README (not run by pytest).

| Script | What it checks | Run |
|---|---|---|
| `compare_final_sdm.py` | `dm_uncertainty` gives the same d_fit, sigma_S, window and fit as the paper's code (`final_sdm.py`) on the paper's FIRES bursts; sigma_DM differs by a few per cent per realisation, since `final_sdm.py` uses the earlier form of V | `SHRINE_SIMS=<shrine_sims/validation_tau02> python compare_final_sdm.py` |
| `mc_covariance.py` | V_jl (`noise_covariance`, paper App. 2) against the covariance of e_par over noise realisations, dedispersed at many trial DMs as the data are; white noise, optionally with self-noise, unequal channels and uneven trial DMs | `python mc_covariance.py [--n_real 5000] [--self_noise 4] [--unequal] [--uneven]` (~15 s per 1000 realisations) |
| `check_delays.py` | `profiles.channel_delays` matches the shift the dedispersion applies to each channel | `python check_delays.py` |
| `mc_voltages.py` | median quoted sigma_DM / scatter of d_fit over noise realisations of a synthetic burst in CRAFT-like voltages | `python mc_voltages.py 0.5 160 [zap_edges]` (burst amplitude, realisations, optional --zap_edges; ~5 min on 2 cores) |
| `make_voltages.py` | synthetic X/Y voltages with a burst dispersed by a known delta-DM (used by the above) | `python make_voltages.py <seed> <dir>` |
| `compare_sn_shrine.py` | `snr.maximise_sn` against SHRINE's `maximise_sn.py` on the same I(DM, t) | `SHRINE=<SHRINE/python> python compare_sn_shrine.py <label> <dt>` |
