import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from util import *
import glob
import os

mat_files = glob.glob('/Users/lixuanxuan/Data/jun13/mat/data/r*alg*.mat')
num_files = len(mat_files)
mask = load_mat('/Users/lixuanxuan/Data/jun13/mat/result/mask.mat')
annulus_mask = make_annulus([401, 401], 80, 200)
for i in range(num_files):
    file = mat_files[i]
    basename, ext = os.path.splitext(file)
    output = basename + '-angular-profile.npy'
    print('processing %d/%d file: %s' %(i+1, num_files, file))
    data = load_mat(file)
    num_image = data.shape[0]
    angular_profiles = []
    for j in range(num_image):
        image = data[j,:,:]
        image = image * mask.T
        image[image<30.] = 0.
        image[image>=30.] = 1.
        # plt.imshow(image)
        # plt.show(block=True)
        angular_profile = calc_angular_profile(image, [200, 200], mask=annulus_mask, mode='sum')
        # plt.plot(angular_profile)
        # plt.show(block=True)
        angular_profiles.append(angular_profile)
    angular_profiles = np.asarray(angular_profiles)
    np.save(output, angular_profiles)