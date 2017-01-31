import numpy as np 
import scipy as sp 
import h5py
from mpi4py import MPI
import os
from pattern2profile import *
from scipy.ndimage.interpolation import rotate
from scipy.signal import savgol_filter, argrelmax
from diffraction_simulation import *
import matplotlib.pyplot as plt


def calc_across_center_line_profile(image, center, angle=0., width=1, mask=None, mode='sum'):
    """Summary
    
    Parameters
    ----------
    image : 2d array
        Input image to calculate angular profile in range of 0 to 180 deg.
    center : array_like with 2 elements
        Center of input image
    angle : float, optional
        Line angle in degrees.
    width : int, optional
        Line width. The default is 1.
    mask : 2d array, optional
        Binary 2d array used in angular profile calculation. The shape must be same with image. 1 means valid while 0 not.
    mode : {'sum', 'mean'}, optional
        'sum'
        By default, mode is 'sum'. This returns the summation of each ring.
    
        'mean'
        Mode 'mean' returns the average value of each ring.
    
    Returns
    -------
    Across center line profile with given width at specified angle: 2d array
        Output array, contains summation or mean value alone the across center line and its indices with respect to the center.
    """
    image = np.asarray(image, dtype=np.float64)
    assert len(image.shape) == 2
    center = np.asarray(center, dtype=np.float64)
    assert center.size == 2
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        assert mask.shape == image.shape
        assert mask.min() >= 0. and mask.max() <= 1.
        mask = (mask > 0.5).astype(np.float64)
    else:
        mask = np.ones_like(image)
    image *= mask 
    # generate a larger image if the given center is not the center of the image.
    sy, sx = image.shape
    if sy % 2 == 0:
        # print('padding along first axis')
        image = np.pad(image, ((0,1), (0,0)), 'constant', constant_values=0)
    if sx % 2 == 0:
        # print('padding along second axis')
        image = np.pad(image, ((0,0), (0,1)), 'constant', constant_values=0)
    sy, sx = image.shape
    if center[0] < sx//2 and center[1] < sy//2:
        # print('case1')
        sx_p = int((sx - center[0]) * 2 - 1)
        sy_p = int((sy - center[1]) * 2 - 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[sy_p-sy:sy_p, sx_p-sx:sx_p] = image
    elif center[0] < sx//2 and center[1] > sy//2:
        # print('case2')
        sx_p = int((sx - center[0]) * 2 - 1)
        sy_p = int((center[1]) * 2 - 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[0:sy, sx_p-sx:sx_p] = image
    elif center[0] > sx//2 and center[1] < sy//2:
        sx_p = int((center[0]) * 2 - 1)
        sy_p = int((sy - center[1]) * 2 - 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[sy_p-sy:sy_p, 0:sx] = image
    else:
        # print('case4')
        sx_p = int((center[0]) * 2 + 1)
        sy_p = int((center[1]) * 2 + 1)
        ex_img = np.zeros((sy_p, sx_p))
        ex_img[0:sy, 0:sx] = image
    rot_img = rotate(ex_img, angle)
    rot_sy, rot_sx = rot_img.shape
    across_line = rot_img[rot_sy//2-width//2:rot_sy//2-width//2+width, :].copy()
    across_line_sum = np.sum(across_line, axis=0)
    line_indices = np.indices(across_line_sum.shape)[0] - rot_sx//2
    line_sum = np.bincount(np.abs(line_indices).ravel(), across_line_sum.ravel())
    if mode == 'sum':
        return line_sum
    elif mode == 'mean':
        line_mean = line_sum.astype(np.float) / width
        return line_mean
    else:
        raise ValueError('Wrong mode: %s' %mode)


def remove_abnormal_spacing(spacing_array):
    while True:
        spacing_array_std = spacing_array.std()
        if spacing_array_std < 1. or spacing_array.size <= 3:
            break
        else:
            spacing_array_mean = spacing_array.mean()
            _dist2mean = np.abs(spacing_array - spacing_array_mean)
            _farmost_index = np.where(_dist2mean==_dist2mean.max())[0][0]
            spacing_array = np.delete(spacing_array, [_farmost_index])
    return spacing_array

if __name__ == '__main__':
    orientation = h5py.File('output/orientation.h5', 'r')
    total_job = orientation['euler_angles'].shape[0]

    model_size = 35
    oversampling_ratio = 7
    core_value = 79
    shell_value = 46
    particle_size = 52E-9
    det_ps = 110E-6
    det_dist = 0.565
    det_size = 401
    src_wavelength = 2.06E-10

    coreshell = Coreshell(model_size=model_size, oversampling_ratio=oversampling_ratio)
    grid_size = cal_grid_size(particle_size=particle_size, det_ps=det_ps, det_dist=det_dist, det_size=det_size,\
                              src_wavelength=src_wavelength, oversampling_ratio=oversampling_ratio)
    intensity3D = np.abs(coreshell.inverse_space) ** 2.

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    job_size = total_job // size
    if rank == size - 1:
        job_idx = np.arange(rank * job_size, total_job)
    else:
        job_idx = np.arange(rank * job_size, (rank + 1) * job_size)
    print('Rank %d processing %d task: %s' % (rank, len(job_idx), str(job_idx)))

    det_mask = np.load('mask.npy')
    mask = make_mask(det_mask=det_mask, inner_radii=90)

    exp_max_angles = []
    exp_spacing_stds = []
    exp_max_intensities = []
    sim_max_angles = []
    sim_spacing_stds = []
    paths = []
    frames = []
    pccs = []
    euler_angles = []
    exp_particle_sizes = []

    for job_id in job_idx:
        path = orientation['paths'][job_id]
        frame = orientation['frames'][job_id]
        pcc = orientation['max_pccs'][job_id]
        euler_angle = orientation['euler_angles'][job_id]
        path = '/Users/lixuanxuan/Data/jun13/mat/data/' + os.path.basename(path)
        exp_pattern = h5py.File(path, 'r')['patterns'][frame].T
        exp_angular_profile = pattern2profile(exp_pattern, mask, log=False)
        exp_max_angle = np.argmax(exp_angular_profile)
        exp_max_intensity = exp_angular_profile[exp_max_angle] / exp_angular_profile.mean()
        exp_across_center_line_profile = calc_across_center_line_profile(exp_pattern, (200, 200), angle=exp_max_angle, width=5, mask=det_mask)
        exp_across_center_line_profile_smoothed = savgol_filter(exp_across_center_line_profile, 15, 3)
        exp_maximum_idx = argrelmax(exp_across_center_line_profile_smoothed, order=11)[0]
        exp_spacing = exp_maximum_idx[1:] - exp_maximum_idx[:-1]
        exp_spacing = remove_abnormal_spacing(exp_spacing)

        sim_pattern = slice2D(intensity3D, euler_angle, grid_size, det_size)
        sim_angular_profile = pattern2profile(sim_pattern, mask, log=False)
        sim_max_angle = np.argmax(sim_angular_profile)
        sim_across_center_line_profile = calc_across_center_line_profile(sim_pattern, (200, 200), angle=exp_max_angle, width=5, mask=det_mask)
        sim_across_center_line_profile_smoothed = savgol_filter(sim_across_center_line_profile, 15, 3)
        sim_maximum_idx = argrelmax(sim_across_center_line_profile_smoothed, order=11)[0]
        sim_spacing = sim_maximum_idx[1:] - sim_maximum_idx[:-1]
        sim_spacing = remove_abnormal_spacing(sim_spacing)

        exp_particle_size = sim_spacing.mean() / exp_spacing.mean() * particle_size
        print('Rank %d, path %s, frame %d, pcc %.2f, euler_angle %s, size %.3e' % (rank, path, frame, pcc, str(euler_angle), exp_particle_size))

        exp_max_angles.append(exp_max_angle)
        exp_spacing_stds.append(exp_spacing.std())
        exp_max_intensities.append(exp_max_intensity)
        sim_max_angles.append(sim_max_angle)
        sim_spacing_stds.append(sim_spacing.std())
        paths.append(path)
        frames.append(frame)
        pccs.append(pcc)
        euler_angles.append(euler_angle)
        exp_particle_sizes.append(exp_particle_size)

    to_h5 = h5py.File('output/size_%d.h5' % rank)
    to_h5.create_dataset('paths', data=paths)
    to_h5.create_dataset('frames', data=frames)
    to_h5.create_dataset('euler_angles', data=euler_angles)
    to_h5.create_dataset('pccs', data=pccs)
    to_h5.create_dataset('exp_particle_sizes', data=exp_particle_sizes)
    to_h5.create_dataset('exp_max_angles', data=exp_max_angles)
    to_h5.create_dataset('exp_spacing_stds', data=exp_spacing_stds)
    to_h5.create_dataset('exp_max_intensities', data=exp_max_intensities)
    to_h5.create_dataset('sim_max_angles', data=sim_max_angles)
    to_h5.create_dataset('sim_spacing_stds', data=sim_spacing_stds)

