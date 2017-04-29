#!/usr/bin/env python

'''Convert 2D diffraction patterns to 1D angular profiles.
Usage:
    pattern2profile.py <pattern_file>... [options]

Options:
    -h --help                               Show this screen.
    --output-dir=output_dir                 Output directory [default: output].
    --apply-mask=apple_mask                 Whether to apply mask [default: False].
    --mask=mask_file                        Mask file in npy format [default: None].
'''

import numpy as np
import glob
import h5py
from mpi4py import MPI
from docopt import docopt
from tqdm import tqdm


def pattern2profile(pattern, mask, binsize=1., log=True, ignore_negative=True):
    pattern = pattern.copy()
    assert pattern.shape[0] == pattern.shape[1]  # must be a square shape
    if ignore_negative:
        pattern[pattern < 0] = 0.
    center = pattern.shape[0] // 2
    pattern *= mask 
    y, x = np.indices((pattern.shape))
    theta = np.rad2deg(np.arctan2(y-center, x-center))
    bin_theta = theta.copy()
    bin_theta[bin_theta<0.] += 180.
    bin_theta = bin_theta / binsize
    bin_theta = np.round(bin_theta).astype(int)
    angular_sum = np.bincount(bin_theta.ravel(), pattern.ravel())  # summation of each ring
    ntheta = np.bincount(bin_theta.ravel(), mask.ravel())
    angular_mean = angular_sum / ntheta
    angular_mean[np.isinf(angular_mean)] = 0.
    angular_mean[np.isnan(angular_mean)] = 0.
    angular_mean /= angular_mean.mean()
    if log:
        angular_mean = np.log(angular_mean + 1.)
    return angular_mean


def make_mask(mask_size=401, inner_radii=75, outer_radii=150, det_mask=None):
    annulus = np.ones((mask_size, mask_size))
    y,x = np.indices((annulus.shape))
    center = mask_size // 2
    r = np.sqrt((x - center)**2. + (y - center)**2.)
    annulus = annulus * (r > inner_radii) * (r < outer_radii)
    if det_mask is None:
        det_mask = np.ones((mask_size, mask_size))
    else:
        assert det_mask.shape == annulus.shape
    return annulus * det_mask


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    pattern_files = argv['<pattern_file>']
    output_dir = argv['--output-dir']
    apply_mask = argv['--apply-mask']
    mask_file = argv['--mask']

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print('Pattern files: %s' % str(pattern_files))
        print('Total pattern files: %d' % len(pattern_files))
        job_size = len(pattern_files) // size
        jobs = []
        for i in range(size):
            if i == (size - 1):
                job = pattern_files[i*job_size:]
            else:
                job = pattern_files[i*job_size:(i+1)*job_size]
            jobs.append(job)
            if i == 0:
                continue
            else:
                comm.send(job, dest=i)
                print('Rank 0 send job to rank %d: %s' % (i, str(job)))
        job = jobs[0]
    else:
        job = comm.recv(source=0)
        print('Rank %d receive job: %s' % (rank, str(job)))
    comm.barrier()

    # Convert to profiles
    if apply_mask == 'True':
        det_mask = np.load(mask_file)
        mask = make_mask(det_mask=det_mask)
    else:
        mask = make_mask(det_mask=None)
    count = 0 
    output = h5py.File('%s/profile_%d.h5' % (output_dir, rank))
    for i in range(len(job)):
        print('===========Rank %d processing %d/%d: %s=============' % (rank, i, len(job)-1, job[i]))
        data = h5py.File(job[i], 'r')
        nb_patterns = data['pattern'].shape[0]
        for j in tqdm(range(nb_patterns)):
            pattern = data['pattern'][j]
            euler_angle = data['euler_angle'][j]
            profile = pattern2profile(pattern, mask, binsize=1.8)
            if count == 0:
                output.create_dataset("profile", data=profile.reshape((1, 101)), maxshape=(None, 101))
                output.create_dataset("euler_angle", data=euler_angle.reshape((1, 3)), maxshape=(None, 3))
            else:
                output['profile'].resize(count+1, axis=0)
                output['euler_angle'].resize(count+1, axis=0)
                output['profile'][count] = profile 
                output['euler_angle'][count] = euler_angle
            count += 1
    output.close()