#!/usr/bin/env python

'''Merge multiple h5 files into single h5.
Usage:
    merge_h5.py <pattern_files>... [-o <output>] 

Options:
    -h --help             Show this screen.
    -o output             Output directory [default: output].
'''


import h5py
import glob
import numpy as np
from docopt import docopt


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    from_h5_files = argv['<pattern_files>']
    output = argv['-o']
    to_h5_file = h5py.File(output)
    N = len(from_h5_files)
    data_dict = {}
    for i in range(N):
        from_h5_file = h5py.File(from_h5_files[i])
        if i == 0:
            for key in from_h5_file.keys():
                shape = from_h5_file[key].shape
                data_dict[key] = from_h5_file[key].value
        else:
            for key in data_dict.keys():
                data_dict[key] = np.append(data_dict[key], from_h5_file[key].value, axis=0)
        from_h5_file.close()
    for key in data_dict.keys():
        to_h5_file.create_dataset(key, data=data_dict[key])
    to_h5_file.close()
