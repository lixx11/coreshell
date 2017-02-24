"""
Calculate max value from radial profile for each pattern in SmallData format.
"""

import numpy as np 
import scipy as sp 
from util import *
import glob
import os
from smallData import SmallData

spiDataDir = '/Users/lixuanxuan/Downloads/test_16bit/'
profile_files = glob.glob('/Users/lixuanxuan/Data/jun13/mat/data/r*alg*-radial-profile.npy')
num_files = len(profile_files)

smallData = SmallData(output='radial-profile-max.h5', smallDataNames=['radial_profile_maxs'])
for i in range(num_files):
    f = profile_files[i]
    basename = os.path.basename(f)
    print('processing %d/%d file: %s' %(i+1, num_files, f))
    spiDataFile = spiDataDir + basename[:-19] + '.h5'
    radial_profiles = np.load(f)
    num_records = radial_profiles.shape[0]
    paths = [spiDataFile] * num_records
    frames = np.arange(0, num_records)
    radial_profile_max = np.max(radial_profiles, axis=1)
    smallData.addRecords(paths, frames, radial_profile_maxs=radial_profile_max.tolist())    

smallData.close()