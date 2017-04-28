#!/usr/bin/env python

'''Evaluate size result for coreshell particle.
Usage:
    evaluate_size.py <size_files>... [options]

Options:
    -h --help                           Show this screen.
    --pcc=pcc                           Min pcc accepted [default: 0.95].
    --max-intensity=max_intensity       Min max intensity accepted [default: 2.5].
    --exp-spacing-std=exp_spacing_std   Max standard derivation of experimental spacing [default: 1.0].
    --sim-spacing-std=sim_spacing_std   Max standard derivation of simulated spacing [default: 1.0].
    --expected-size=expected_size       Expected particle size in nanometer[default: 52].
    --diff-angle=diff_angle             Max angle difference between simulated and experimental pattern [default: 3].
    --upper-limit=upper_limit           Upper limit ratio of particle size against expected [default: 1.5].
    --lower-limit=lower_limit           Lower limit ratio of particle size against expected [default: 0.5].
    --output=output_dir                 Output filename [default: size.txt].
'''

import h5py
import numpy as np 
import glob
from docopt import docopt


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    size_files = argv['<size_files>']
    expected_size = float(argv['--expected-size']) * 1E-9
    min_pcc = float(argv['--pcc'])
    min_max_intensity = float(argv['--max-intensity'])
    max_exp_spacing_std = float(argv['--exp-spacing-std'])
    max_sim_spacing_size = float(argv['--sim-spacing-std'])
    diff_angle = float(argv['--diff-angle'])
    upper_limit = float(argv['--upper-limit'])
    lower_limit = float(argv['--lower-limit'])
    output = argv['--output']

    # do evaluation and collection
    valid_sizes = []
    for f in size_files:
        h5f = h5py.File(f, 'r')
        pccs = h5f['pccs']
        exp_particle_sizes = h5f['exp_particle_sizes']
        exp_max_angles = h5f['exp_max_angles']
        exp_spacing_stds = h5f['exp_spacing_stds']
        exp_max_intensities = h5f['exp_max_intensities']
        sim_max_angles = h5f['sim_max_angles']
        sim_spacing_stds = h5f['sim_spacing_stds']
        for i in range(len(pccs)):
            if pccs[i] > min_pcc \
                and exp_max_intensities[i] > min_max_intensity \
                and exp_spacing_stds[i] < max_exp_spacing_std \
                and (exp_max_angles[i] - sim_max_angles[i]) < diff_angle \
                and sim_spacing_stds[i] < max_sim_spacing_size \
                and exp_particle_sizes[i] < expected_size * upper_limit \
                and exp_particle_sizes[i] > expected_size * lower_limit:
                valid_sizes.append(exp_particle_sizes[i])
    valid_sizes = np.asarray(valid_sizes)
    np.savetxt(output, valid_sizes)