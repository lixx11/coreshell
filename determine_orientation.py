import numpy as np
import glob
import h5py
from mpi4py import MPI
from pattern2profile import *
from scipy.stats import pearsonr


if __name__ == '__main__':
    output_dir = 'output'
    reference = h5py.File('/Users/lixuanxuan/Downloads/profile.h5', 'r')
    ref_profiles = reference['profile'].value
    ref_euler_angle = reference['euler_angle'].value

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        exp_files = glob.glob('/Users/lixuanxuan/Data/jun13/mat/data/r*alg*.mat')
        print('Experimental data: %s' % str(exp_files))
        print('Total experiment files: %d' % len(exp_files))

        job_size = len(exp_files) // size
        jobs = []
        for i in range(size):
            if i == (size - 1):
                job = exp_files[i*job_size:]
            else:
                job = exp_files[i*job_size : (i+1)*job_size]
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

    det_mask = np.load('mask.npy')
    mask = make_mask(det_mask=det_mask)
    h5f = h5py.File('%s/orientation_%d.h5' % (output_dir, rank))
    paths = []
    frames = []
    euler_angles = []
    max_pccs = []
    for i in range(len(job)):
        print('===========Rank %d processing %d/%d: %s=============' % (rank, i, len(job)-1, job[i]))
        data = h5py.File(job[i], 'r')['patterns']
        for j in range(data.shape[0]):
            pattern = data[j].T 
            profile = pattern2profile(pattern, mask, binsize=1.8)
            # find the max pcc
            a = profile 
            b = ref_profiles
            N_ref = ref_profiles.shape[0]
            pccs = ((a-a.mean())*(b-b.mean(axis=1).reshape((N_ref,1)))).sum(axis=1)/np.sqrt(((a-a.mean())**2).sum()*((b-b.mean(axis=1).reshape((N_ref,1)))**2).sum(axis=1))
            max_id = np.argmax(pccs)
            max_pcc = pccs[max_id]
            euler_angle = ref_euler_angle[max_id]
            print('Rank %d, %s, frame %d/%d, euler angle: %s, pcc: %.3f' % (rank, job[i], j, data.shape[0], euler_angle, max_pcc))
            paths.append(job[i])
            frames.append(j)
            euler_angles.append(euler_angle)
            max_pccs.append(max_pcc)
    h5f.create_dataset('paths', data=paths)
    h5f.create_dataset('frames', data=frames)
    h5f.create_dataset('euler_angles', data=euler_angles)
    h5f.create_dataset('max_pccs', data=max_pccs)

