"""
find absolute angles and relative angles from angular profiles.
"""

import numpy as np 
import scipy as sp 
from scipy.signal import argrelmax
from util import *
import matplotlib.pyplot as plt 
import glob
import os


def calc_relative_angles(abs_angles):
    abs_angles = np.asarray(abs_angles)
    if abs_angles.size == 1:
        return [0,]
    rel_angles = []
    abs_angles = np.sort(abs_angles)
    for i in range(abs_angles.size):
        for j in range(abs_angles.size):
            if j>i:
                rel_angles.append(abs_angles[j]-abs_angles[i])
    return rel_angles


npy_files = glob.glob('/Users/lixuanxuan/Data/jun13/mat/data/r*alg*angular-profile.npy')
num_files = len(npy_files)
threshold = 2.

for i in range(num_files):
    file = npy_files[i]
    basename, ext = os.path.splitext(file)
    output = basename + '-features.npz'
    print('processing %d/%d file: %s' %(i+1, num_files, file))
    profiles = load_npy(file)
    abs_peak_angles = [] # absolute peak found in angular profile
    rel_peak_angles = [] # relative peak computed from absolute peak
    for j in range(profiles.shape[0]):
        profile = profiles[j,:]
        profile_with_noise = profile + np.random.rand(profile.size)*1E-5  # add some noise to avoid same integer value in profile
        maxima_indices = argrelmax(profile_with_noise, order=11)[0]
        maximas = profile[maxima_indices]
        filtered_maxima_indices = maxima_indices[maximas > threshold*profile.mean()]
        filtered_maximas = profile[filtered_maxima_indices]

        rel_angles = calc_relative_angles(filtered_maxima_indices)
        abs_peak_angles.append(filtered_maxima_indices)
        rel_peak_angles.append(rel_angles)
    np.savez(output, absolute_angles=np.asarray(abs_peak_angles),
                    relative_angles=np.asarray(rel_peak_angles))
        # plt.plot(profile)
        # print(filtered_maxima_indices)
        # print(rel_angles)
        # plt.show(block=True)