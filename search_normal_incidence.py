import numpy as np 
import scipy as sp 
from scipy.signal import argrelmax
from util import *
import matplotlib.pyplot as plt 
import glob
import os


def get_all_pairs(array):
    assert len(array.shape) == 1
    array = np.sort(array)
    pairs = []
    for i in range(array.size):
        for j in range(array.size):
            if j > i:
                pairs.append([array[i], array[j]])
    return pairs

npz_files = glob.glob('/Users/lixuanxuan/Data/jun13/mat/data/r*alg*angular-profile-features.npz')
num_files = len(npz_files)
tolerance = 5.  # 90+-10deg 
max_intensity_ratio = 1.5  # normal incidence should have 2 peaks in angular profile, which has similar intensity.

count = 0
normal_incidence_images = []
for i in range(num_files):
    file = npz_files[i]
    basename, ext = os.path.splitext(file)
    matfile = basename[:-25] + '.mat'
    profilefile = basename[:-9] + '.npy'
    output = basename[:-25] + '.txt'

    print('processing %d/%d file: %s' %(i+1, num_files, file))
    output_f = open(output, 'w')
    # print('mat file: %s' %matfile)
    # print('angular profile file: %s' %profilefile)
    data = np.load(file)
    matdata = load_mat(matfile)
    profiledata = np.load(profilefile)
    absolute_angles = data['absolute_angles']
    relative_angles = data['relative_angles']
    for j in range(len(relative_angles)):
        abs_angles = np.asarray(absolute_angles[j])
        rel_angles = np.asarray(relative_angles[j])
        image = np.asarray(matdata[j,:,:], dtype=np.int)
        angular_profile = np.asarray(profiledata[j,:])
        if rel_angles.size != 1:
            continue
        if np.abs(rel_angles - 90.).min() < tolerance:
            angle_pairs = get_all_pairs(abs_angles)
            perpendicular_pairs = []
            for k in range(len(angle_pairs)):
                angle_pair = angle_pairs[k]
                rel_angle = angle_pair[1] - angle_pair[0]
                if abs(rel_angle - 90.) < tolerance:
                    perpendicular_pairs.append(angle_pair)
            for k in range(len(perpendicular_pairs)):
                angle_pair = perpendicular_pairs[k]
                intensity_pair = angular_profile[angle_pair]
                if max(intensity_pair) / min(intensity_pair) <= max_intensity_ratio:
                    normal_incidence_images.append(image)
                    count += 1
                    output_f.write("%s %d\n" %(matfile, j))
                    break
    output_f.close()
    print('current count: %d' %count)
print(count)