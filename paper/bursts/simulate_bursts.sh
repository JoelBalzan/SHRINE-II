#!/bin/bash
# Regenerate the noiseless bursts of the paper with FIRES (https://github.com/JoelBalzan/FIRES; run in
# an environment where the fires command is installed). The bursts are already in this folder as
# <tag>.npz, so this is needed only to change them.
#   single, double      the validation bursts (Section 2.1), configs/fires_{single,double}.toml
#   drift000 ... 080    the drifting sub-bursts (Section 5.1), configs/fires_drift*.toml
# FIRES writes out_<tag>/val_<tag>_dspec/{out_I,freq,time}.npy (dynamic spectrum at the true DM,
# channels from the top of the band down, MHz, ms); these are packed into <tag>.npz. FIRES then reports
# "Plot mode None is not defined", which is harmless.
cd "$(dirname "$0")"
for config in configs/fires_single.toml configs/fires_double.toml configs/fires_drift*.toml; do
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
