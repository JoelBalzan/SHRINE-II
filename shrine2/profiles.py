"""I(d_j, t): frequency-summed profiles at each trial DM, from CRAFT X/Y voltages, as SHRINE's
generate_profiles.py.

The steps, per trial DM (SHRINE's do_DM), are unchanged from SHRINE:
  1. coherently dedisperse the X and Y voltages by the trial DM (relative to the DM they were
     already dedispersed to);
  2. form a dynamic spectrum of nchan = 336 coarse channels, one FFT per 336 voltage samples
     (1 us at 336 MHz);
  3. I = |X|^2 + |Y|^2;
  4. leave out the zapped channels (--zap, --zap_edges);
  5. normalise each remaining channel to zero mean and unit standard deviation over the crop;
  6. sum them;
  7. average down to dt microseconds (sum of dt samples / sqrt(dt)).
They are written out explicitly, and vectorised. SHRINE always left out the last channel and
zapped nothing else; --zap 335 gives SHRINE's profiles exactly.

channel_delays() gives the dispersion delay and noise share of each summed channel, which the
covariance (Appendix 2) needs, so it must be given the same zapped channels.
"""
import numpy as np

# The dispersion constant used by SHRINE, MHz^-2 pc cm^-3 s^-1 (1 / 4.149e3 s MHz^2).
K_DM = 2.41e-4

# Coarse channels formed by SHRINE: 1 MHz each over 336 MHz.
NCHAN = 336

# CRAFT voltage sample interval, us (336 MHz bandwidth).
US_PER_SAMPLE = 1 / 336


def fine_freqs(f0, bw, n):
    """Frequency of each FFT bin of an n-sample voltage series (SHRINE's get_freqs), from the top
    of the band down.

    Args:
        f0: centre frequency, MHz.
        bw: bandwidth, MHz.
        n: number of voltage samples (= FFT bins).
    Returns:
        frequency of each bin, shape (n,), MHz.
    """
    f_max = f0 + bw / 2
    f_min = f0 - bw / 2
    bin_width = bw / n                                   # width of one FFT bin, MHz

    # n bins running down from the top of the band, offset by half a bin (as SHRINE).
    return np.linspace(f_max, f_min, n, endpoint=False) + bin_width / 2


def crop(X, Y, I, crop_ms, peak_ms=None):
    """Cut crop_ms milliseconds of voltages centred on the burst (SHRINE's do_crop).

    The burst is at the brightest 1-ms bin of I summed over channels, unless peak_ms is given.

    Args:
        X, Y: complex voltages, shape (n,), one sample per 1/336 us.
        I: intensity, shape (n_chan, n_t) at 1 us, used only to find the burst.
        crop_ms: length of the crop, ms.
        peak_ms: time of the burst in ms (--force_peak), or None to find it from I.
    Returns:
        X, Y cropped (copies).
    """
    if peak_ms is None:
        # Sum over channels and find the brightest 1-ms bin. As SHRINE, the last column of I (the
        # last time sample, for I of shape (n_chan, n_t)) is left out.
        I_t = I[:, :-1].sum(axis=0)
        peak_ms = int(np.argmax(tscrunch(I_t, 1000)))

    # The crop in voltage samples: crop_ms centred on the peak.
    samples_per_ms = 1000 * NCHAN
    mid = peak_ms * samples_per_ms
    half = crop_ms * samples_per_ms // 2
    return X[mid - half:mid + half].copy(), Y[mid - half:mid + half].copy()


def tscrunch(a, n):
    """Average down by n samples along axis 0, as SHRINE: sum of n samples / sqrt(n) (keeps the
    white-noise variance).

    Args:
        a: array with time along axis 0.
        n: number of samples averaged into each output sample.
    Returns:
        the averaged array, a.shape[0] // n samples along axis 0.
    """
    # Drop the samples that do not fill a whole output sample, then sum each group of n.
    m = a.shape[0] // n
    groups = a[:m * n].reshape(m, n, *a.shape[1:])
    return groups.sum(axis=1) / np.sqrt(n)


def dedispersion_phase(dm, freqs):
    """Coherent dedispersion: the phase that removes a DM dm (pc cm^-3) from a voltage spectrum
    (SHRINE's dedisperse).

    exp(2 pi i dm/K_DM (f - f_ref)^2 / (f_ref^2 f)), with f in MHz (x 1e6 for the units) and f_ref
    the median frequency (Lorimer & Kramer 2005).

    Args:
        dm: DM to remove, pc cm^-3, relative to the DM the voltages were already dedispersed to.
        freqs: frequency of each FFT bin, MHz (fine_freqs).
    Returns:
        complex phase for each bin, shape (n,).
    """
    f_ref = np.median(freqs)

    # i times the phase, in radians.
    i_phase = 2j * np.pi * dm / K_DM * (freqs - f_ref)**2 / (f_ref**2 * freqs) * 1e6
    return np.exp(i_phase)


def dynamic_spectrum(v):
    """Coarse channels of a voltage series (SHRINE's generate_dynspec): one FFT per NCHAN samples.

    Args:
        v: complex voltages, shape (n,).
    Returns:
        coarse channels, shape (n // NCHAN, NCHAN), complex64: (time in 1-us samples, channel).
    """
    # Cut the series into blocks of NCHAN samples (dropping any remainder) and FFT each block.
    n_t = v.size // NCHAN
    blocks = v[:n_t * NCHAN].reshape(n_t, NCHAN)
    return np.fft.fft(blocks, axis=1).astype(np.complex64)


def profile(FX, FY, dm, freqs, dt, zap=()):
    """The frequency-summed profile at one trial DM (SHRINE's do_DM), from the voltage spectra.

    Args:
        FX, FY: FFTs of the cropped X and Y voltages, shape (n,).
        dm: trial delta-DM, pc cm^-3.
        freqs: frequency of each FFT bin, MHz (fine_freqs).
        dt: output time resolution, us.
        zap: indices of the coarse channels to leave out (zapped_channels).
    Returns:
        the profile x at this trial DM, shape (n // NCHAN // dt,).
    """
    phase = dedispersion_phase(dm, freqs)

    # Steps 1-3: dedisperse each polarisation, form its coarse channels and add the powers,
    # I(t, channel) = |X|^2 + |Y|^2.
    X_power = np.abs(dynamic_spectrum(np.fft.ifft(FX * phase)))**2
    Y_power = np.abs(dynamic_spectrum(np.fft.ifft(FY * phase)))**2
    I_tc = X_power + Y_power

    # Step 4: leave out the zapped channels.
    kept = np.setdiff1d(np.arange(NCHAN), zap)
    I_tc = I_tc[:, kept]

    # Step 5: each channel to zero mean and unit standard deviation over the crop.
    I_tc = (I_tc - I_tc.mean(axis=0)) / I_tc.std(axis=0)

    # Steps 6 and 7: sum the channels, then average down to dt us.
    I_t = I_tc.sum(axis=1)
    return tscrunch(I_t, dt)


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

    # The voltage spectra are computed once and reused at every trial DM.
    FX = np.fft.fft(X)
    FY = np.fft.fft(Y)

    rows = [profile(FX, FY, dm, freqs, dt, zap) for dm in dms]
    return np.array(rows)


def zapped_channels(zap=(), zap_edges=0, nchan=NCHAN):
    """The coarse channels to leave out, from --zap and --zap_edges.

    Channel c is centred on f_max - c bw/nchan (see channel_delays), so channels 1, 2, ... run down
    from the top of the band and nchan-1, nchan-2, ... up from the bottom; channel 0 straddles both
    edges (half at each).

    Args:
        zap: channel indices and inclusive ranges, as strings, e.g. ["0", "100-110", "335"] or
            ["0,100-110,335"].
        zap_edges: leave out the first and last zap_edges channel indices, 0 .. zap_edges-1 and
            nchan-zap_edges .. nchan-1 (1 zaps channels 0 and nchan-1). Because channel 0 holds
            half a channel at each edge, this removes zap_edges - 1/2 channels' width from the top
            and zap_edges + 1/2 from the bottom.
        nchan: number of coarse channels.
    Returns:
        sorted array of channel indices.
    Raises:
        ValueError if a channel lies outside 0 .. nchan-1 or an item is not an index or range.
    """
    out = set()

    # Indices ("335") and inclusive ranges ("100-110"), given separately or comma-separated.
    for item in ",".join(zap).split(","):
        if not item.strip():
            continue
        first, _, last = item.partition("-")
        if not last:
            last = first
        out.update(range(int(first), int(last) + 1))

    # The first and the last zap_edges channel indices.
    out.update(range(zap_edges))
    out.update(range(nchan - zap_edges, nchan))

    for c in out:
        if c < 0 or c >= nchan:
            raise ValueError(f"zapped channels must lie in 0 .. {nchan - 1}")
    return np.array(sorted(out), dtype=int)


def channel_delays(f0, bw=336.0, nchan=NCHAN, zap=()):
    """Dispersion delay per unit DM of each channel summed by profile(), and its share of the noise.

    Coarse channel c is FFT bin c of a block of nchan voltage samples; SHRINE's FFT bins run down
    from the top of the band, so channel c is centred on f_max - c bw/nchan. Channel 0 is centred
    on f_max itself: its upper half lies above the band and wraps round to the bottom, so it holds
    the top and bottom half-channels and enters as two halves of half a share each. Delays are
    relative to f0:
        tau_nu = (1/K_DM) (1/f_nu^2 - 1/f0^2)   us per pc cm^-3, with f in MHz,
    with the sign convention I(t; d) = sum_nu I_nu(t + d tau_nu) (App. 2, Eq. epchan): a positive
    trial DM moves low frequencies earlier and high frequencies later.

    Args:
        f0: centre frequency, MHz (the delays are relative to it).
        bw: bandwidth, MHz.
        nchan: number of coarse channels formed.
        zap: indices of the channels left out (as given to make_profiles).
    Returns:
        tau: delay per unit DM of each summed (half-)channel, us per pc cm^-3; divide by dt (us)
            for dt_nu in samples.
        share: each entry's share of a full channel's noise variance (0.5 for the halves of
            channel 0, else 1).
    """
    f_max = f0 + bw / 2
    f_min = f0 - bw / 2
    chan_width = bw / nchan                              # width of one coarse channel, MHz

    # The centre frequency and noise share of each summed channel (two halves for channel 0).
    f = []
    share = []
    for c in np.setdiff1d(np.arange(nchan), zap):
        if c == 0:
            f.append(f_max - chan_width / 4)             # top half-channel
            f.append(f_min + chan_width / 4)             # bottom half-channel
            share.append(0.5)
            share.append(0.5)
        else:
            f.append(f_max - c * chan_width)
            share.append(1.0)
    f = np.array(f)
    share = np.array(share)

    # Delay relative to f0, us per pc cm^-3 (1e6 converts s to us).
    tau = 1e6 / K_DM * (1 / f**2 - 1 / f0**2)
    return tau, share


def one_sample_dm_step(f0, bw, dt, nchan=NCHAN, zap=()):
    """The DM step that moves the bottom of the band by one output sample (dt us) relative to the
    top.

    This is the DM resolution set by the time and frequency coverage: dt / (tau_bottom - tau_top),
    with tau the dispersion delay per unit DM of the channels summed (channel_delays). Finer steps
    still change the profile, by fractions of a sample, and the fit may need them when sigma_DM is
    smaller than this (optimise_dm.py refines the trial DMs to sigma_DM/6 when it is).

    Args:
        f0, bw: centre frequency and bandwidth, MHz.
        dt: output time resolution, us.
        nchan: number of coarse channels formed.
        zap: indices of the channels left out.
    Returns:
        the step, pc cm^-3.
    """
    tau, _ = channel_delays(f0, bw, nchan, zap)
    delay_spread = tau.max() - tau.min()
    return dt / delay_spread
