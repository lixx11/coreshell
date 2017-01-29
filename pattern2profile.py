import numpy as np
import glob
import h5py


def pattern2profile(pattern, mask, binsize=1., log=True, ignore_negative=True):
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
    if log:
        angular_mean = np.log(np.abs(angular_mean)+1.)
    # normalization
    angular_mean /= angular_mean.max()
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
    data_dir = 'output'
    data_list = glob.glob(data_dir + '/data*.h5')
    output_dir = 'output'
    det_mask = np.load('mask.npy')
    mask = make_mask(det_mask=det_mask)

    h5f = h5py.File(output_dir + '/profile.h5')
    for f in data_list:
        data = h5py.File(f, 'r')
        N_pattern = data['pattern'].shape[0]
        for i in range(N_pattern):
            print i
            pattern = data['pattern'][i]
            euler_angle = data['euler_angle'][i]
            profile = pattern2profile(pattern, mask, binsize=1.8)
            if i == 0:
                h5f.create_dataset("profile", data=profile.reshape((1, 101)), maxshape=(None, 101))
                h5f.create_dataset("euler_angle", data=euler_angle.reshape((1, 3)), maxshape=(None, 3))
            else:
                h5f['profile'].resize(i+1, axis=0)
                h5f['euler_angle'].resize(i+1, axis=0)
                h5f['profile'][i] = profile 
                h5f['euler_angle'][i] = euler_angle
    h5f.close()

    