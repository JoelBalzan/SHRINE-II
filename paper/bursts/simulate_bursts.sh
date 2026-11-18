#!/bin/bash
# Regenerate the noiseless bursts of the paper with FIRES (https://github.com/JoelBalzan/FIRES; run in
# an environment where the fires command is installed). The bursts are already in this folder as
# <tag>.npz, so this is needed only to change them.
#   single, double      the validation bursts (Section 2.1), configs/fires_{single,double}.toml
#   drift000 ... 080    the drifting sub-bursts (Section 5.2), configs/fires_drift*.toml
#   tau000 ... tau040   single and double with tau = 0, 0.05, 0.1, 0.2, 0.4 ms at 1 GHz (Section 5.1,
#                       sec5_discussion/tau_scan.py), configs/fires_tau*_{single,double}.toml
# FIRES writes out_<tag>/val_<tag>_dspec/{out_I,freq,time}.npy (dynamic spectrum at the true DM,
# channels from the top of the band down, MHz, ms); these are packed into <tag>.npz. FIRES then reports
# "Plot mode None is not defined", which is harmless.
# RM-Tools (imported by FIRES) imports matplotlib's Tk backend; where Python has no Tk, put stub
# tkinter and matplotlib.backends.backend_tkagg modules on PYTHONPATH and set MPLBACKEND=Agg.
cd "$(dirname "$0")"
for config in configs/fires_single.toml configs/fires_double.toml configs/fires_drift*.toml configs/fires_tau*.toml; do
    tag=$(basename "$config" .toml | sed 's/fires_//')
    fires --config-dir "$config" -f "val_$tag" -o "out_$tag" --save-dspec -p None \
          --override-plot general.show_plots=false general.save_plots=false
    python3 -c "
import numpy as np
d = 'out_$tag/val_${tag}_dspec/'
np.savez_compressed('$tag.npz', I=np.load(d + 'out_I.npy'), freq=np.load(d + 'freq.npy'),
                    time=np.load(d + 'time.npy'))
"
done
