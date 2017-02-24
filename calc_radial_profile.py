import numpy as np 
import scipy as sp 
from util import *
import glob
import os

mat_files = glob.glob('/Users/lixuanxuan/Data/jun13/mat/data/r*alg*.mat')
num_files = len(mat_files)
mask = load_data('/Users/lixuanxuan/Data/jun13/mat/result/mask.mat', 'mask')
for i in range(num_files):
    file = mat_files[i]
    basename, ext = os.path.splitext(file)
    output = basename + '-radial-profile.npy'
    print('processing %d/%d file: %s' %(i+1, num_files, file))
    data = load_data(file, 'patterns')
    num_image = data.shape[0]
    radial_profiles = []
    for j in range(num_image):
        image = data[j,:,:].T
        masked_image = image * mask 
        radial_profile = calc_radial_profile(masked_image, [200, 200], mask=mask, mode='mean')
        radial_profiles.append(radial_profile)
    radial_profiles = np.asarray(radial_profiles)
    np.save(output, radial_profiles)