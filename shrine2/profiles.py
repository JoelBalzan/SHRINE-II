"""I(d_j, t): frequency-summed profiles at each trial DM, from CRAFT X/Y voltages, as SHRINE's generate_profiles.py.

The steps, per trial DM (SHRINE's do_DM), are unchanged from SHRINE:
  1. coherently dedisperse the X and Y voltages by the trial DM (relative to the DM they were already dedispersed to);
  2. form a dynamic spectrum of nchan = 336 coarse channels, one FFT per 336 voltage samples (1 us at 336 MHz);
  3. I = |X|^2 + |Y|^2;
  4. leave out the zapped channels (--zap, --zap_edges);
  5. normalise each remaining channel to zero mean and unit standard deviation over the crop;
  6. sum them;
  7. average down to dt microseconds (sum of dt samples / sqrt(dt)).
They are written out explicitly, and vectorised. SHRINE always left out the last channel and zapped nothing else;
--zap 335 gives SHRINE's profiles exactly.

channel_delays() gives the dispersion delay and noise share of each summed channel, which the covariance
(Appendix 2) needs, so it must be given the same zapped channels.
"""
import numpy as np

K_DM = 2.41e-4          # MHz^-2 pc cm^-3 s^-1, the dispersion constant used by SHRINE (1 / 4.149e3 s MHz^2)
NCHAN = 336             # coarse channels formed by SHRINE (1 MHz each over 336 MHz)
US_PER_SAMPLE = 1 / 336  # CRAFT voltage sample interval in us (336 MHz bandwidth)


def fine_freqs(f0, bw, n):
    """Frequency of each FFT bin of an n-sample voltage series (SHRINE's get_freqs), from the top of the band down.

    Args:
        f0: centre frequency, MHz.
        bw: bandwidth, MHz.
        n: number of voltage samples (= FFT bins).
    Returns:
        frequency of each bin, shape (n,), MHz.
    """
    bin_width = bw / n                                   # width of one FFT bin, MHz
    return np.linspace(f0 + bw / 2, f0 - bw / 2, n, endpoint=False) + bin_width / 2


def crop(X, Y, I, crop_ms, peak_ms=None):
    """Cut crop_ms milliseconds of voltages centred on the burst (SHRINE's do_crop).

    The peak is the brightest 1-ms bin of I (channels summed, last channel left out), unless peak_ms is given.

    Args:
        X, Y: complex voltages, shape (n,), one sample per 1/336 us.
        I: intensity, shape (n_chan, n_t) at 1 us, used only to find the burst.
        crop_ms: length of the crop, ms.
        peak_ms: time of the burst in ms (--force_peak), or None to find it from I.
    Returns:
        X, Y cropped (copies).
    """
    if peak_ms is None:
        I_t = I[:, :-1].sum(axis=0)
        peak_ms = int(np.argmax(tscrunch(I_t, 1000)))
    per_ms = 1000 * NCHAN                                # voltage samples per ms
    mid, half = peak_ms * per_ms, crop_ms * per_ms // 2
    return X[mid - half:mid + half].copy(), Y[mid - half:mid + half].copy()


def tscrunch(a, n):
    """Average down by n samples along axis 0, as SHRINE: sum of n samples / sqrt(n) (keeps white-noise variance).

    Args:
        a: array with time along axis 0.
        n: number of samples averaged into each output sample.
    Returns:
        the averaged array, a.shape[0] // n samples along axis 0.
    """
    m = a.shape[0] // n
    return a[:m * n].reshape(m, n, *a.shape[1:]).sum(axis=1) / np.sqrt(n)


def dedispersion_phase(dm, freqs):
    """Coherent dedispersion: the phase that removes a DM dm (pc cm^-3) from a voltage spectrum (SHRINE's dedisperse).

    exp(2 pi i dm/K_DM (f - f_ref)^2 / (f_ref^2 f)), with f in MHz (x 1e6 for the units) and f_ref the median
    frequency (Lorimer & Kramer 2005).

    Args:
        dm: DM to remove, pc cm^-3, relative to the DM the voltages were already dedispersed to.
        freqs: frequency of each FFT bin, MHz (fine_freqs).
    Returns:
        complex phase for each bin, shape (n,).
    """
    f_ref = np.median(freqs)
    return np.exp(2j * np.pi * dm / K_DM * (freqs - f_ref)**2 / (f_ref**2 * freqs) * 1e6)


def dynamic_spectrum(v):
    """Coarse channels of a voltage series (SHRINE's generate_dynspec): one FFT per NCHAN samples -> (time, channel).

    Args:
        v: complex voltages, shape (n,).
    Returns:
        coarse channels, shape (n // NCHAN, NCHAN), complex64: (time in 1-us samples, channel).
    """
    n_t = v.size // NCHAN
    return np.fft.fft(v[:n_t * NCHAN].reshape(n_t, NCHAN), axis=1).astype(np.complex64)


def profile(FX, FY, dm, freqs, dt, zap=()):
    """The frequency-summed profile at one trial DM (SHRINE's do_DM), from the voltage spectra FX, FY.

    Args:
        FX, FY: FFTs of the cropped X and Y voltages, shape (n,).
        dm: trial delta-DM, pc cm^-3.
        freqs: frequency of each FFT bin, MHz (fine_freqs).
        dt: output time resolution, us.
        zap: indices of the coarse channels to leave out (zapped_channels).
    Returns:
        the profile x at this trial DM, shape (n // NCHAN // dt,).
    """
    ph = dedispersion_phase(dm, freqs)
    I_tc = (np.abs(dynamic_spectrum(np.fft.ifft(FX * ph)))**2
            + np.abs(dynamic_spectrum(np.fft.ifft(FY * ph)))**2)       # I(t, channel)
    I_tc = I_tc[:, np.setdiff1d(np.arange(NCHAN), zap)]                 # leave out the zapped channels
    I_tc = (I_tc - I_tc.mean(axis=0)) / I_tc.std(axis=0)                 # each channel: zero mean, unit std
    return tscrunch(I_tc.sum(axis=1), dt)                                # sum the channels, to dt us


def make_profiles(X, Y, dms, f0, bw, dt, zap=()):
    """I(d_j, t) for the trial DMs dms (one row per trial DM).

    Args:
        X, Y: cropped complex voltages, shape (n,).
        dms: trial delta-DMs, shape (n_d,), pc cm^-3.
        f0, bw: centre frequency and bandwidth, MHz.
        dt: output time resolution, us.
        zap: indices of the coarse channels to leave out (zapped_channels).
    Returns:
        I(d_j, t), shape (n_d, N).
    """
    freqs = fine_freqs(f0, bw, X.shape[0])
    FX, FY = np.fft.fft(X), np.fft.fft(Y)                # voltage spectra, computed once
    return np.array([profile(FX, FY, dm, freqs, dt, zap) for dm in dms])


def zapped_channels(zap=(), zap_edges=0, nchan=NCHAN):
    """The coarse channels to leave out, from --zap and --zap_edges.

    Channel c is centred on f_max - c bw/nchan (see channel_delays), so channels 1, 2, ... run down from the top
    of the band and nchan-1, nchan-2, ... up from the bottom; channel 0 straddles both edges (half at each).

    Args:
        zap: channel indices and inclusive ranges, as strings, e.g. ["0", "100-110", "335"] or ["0,100-110,335"].
        zap_edges: leave out the first and last zap_edges channel indices, 0 .. zap_edges-1 and
            nchan-zap_edges .. nchan-1 (1 zaps channels 0 and nchan-1). Because channel 0 holds half a channel at
            each edge, this removes zap_edges - 1/2 channels' width from the top and zap_edges + 1/2 from the bottom.
        nchan: number of coarse channels.
    Returns:
        sorted array of channel indices.
    """
    out = set()
    for item in ",".join(zap).split(","):
        if not item.strip():
            continue
        lo, _, hi = item.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    out.update(range(zap_edges))                         # first zap_edges channels
    out.update(range(nchan - zap_edges, nchan))          # last zap_edges channels
    if any(c < 0 or c >= nchan for c in out):
        raise ValueError(f"zapped channels must lie in 0 .. {nchan - 1}")
    return np.array(sorted(out), dtype=int)


def channel_delays(f0, bw=336.0, nchan=NCHAN, zap=()):
    """Dispersion delay per unit DM of each channel summed by profile(), and its share of the noise.

    Coarse channel c is FFT bin c of a block of nchan voltage samples; SHRINE's FFT bins run down from the top of
    the band, so channel c is centred on f_max - c bw/nchan. Channel 0 is centred on f_max itself: its upper half
    lies above the band and wraps round to the bottom, so it holds the top and bottom half-channels and enters as
    two halves of half a share each. Delays are relative to f0:
        tau_nu = (1/K_DM) (1/f_nu^2 - 1/f0^2)   us per pc cm^-3, with f in MHz,
    with the sign convention I(t; d) = sum_nu I_nu(t + d tau_nu) (App. 2, Eq. epchan): a positive trial DM moves
    low frequencies earlier and high frequencies later.

    Args:
        f0: centre frequency, MHz (the delays are relative to it).
        bw: bandwidth, MHz.
        nchan: number of coarse channels formed.
        zap: indices of the channels left out (as given to make_profiles).
    Returns:
        tau: delay per unit DM of each summed (half-)channel, us per pc cm^-3; divide by dt (us) for dt_nu in samples.
        share: each entry's share of a full channel's noise variance (0.5 for the halves of channel 0, else 1).
    """
    f_max, f_min = f0 + bw / 2, f0 - bw / 2
    chan_width = bw / nchan                              # width of one coarse channel, MHz
    f, share = [], []
    for c in np.setdiff1d(np.arange(nchan), zap):
        if c == 0:                                       # straddles the band edge: top and bottom half-channels
            f += [f_max - chan_width / 4, f_min + chan_width / 4]
            share += [0.5, 0.5]
        else:
            f.append(f_max - c * chan_width)
            share.append(1.0)
    f, share = np.array(f), np.array(share)
    tau = 1e6 / K_DM * (1 / f**2 - 1 / f0**2)
    return tau, share


def one_sample_dm_step(f0, bw, dt, nchan=NCHAN, zap=()):
    """The DM step that moves the bottom of the band by one output sample (dt us) relative to the top.

    This is the DM resolution set by the time and frequency coverage: dt / (tau_bottom - tau_top), with tau the
    dispersion delay per unit DM of the channels summed (channel_delays). Finer steps still change the profile,
    by fractions of a sample, and the fit may need them when sigma_DM is smaller than this (optimise_dm.py
    refines the trial DMs to sigma_DM/6 when it is).

    Args:
        f0, bw: centre frequency and bandwidth, MHz.
        dt: output time resolution, us.
        nchan: number of coarse channels formed.
        zap: indices of the channels left out.
    Returns:
        the step, pc cm^-3.
    """
    tau, _ = channel_delays(f0, bw, nchan, zap)
    return dt / (tau.max() - tau.min())
