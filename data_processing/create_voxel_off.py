from functools import partial
from .voxels import VoxelGrid
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse

def create_voxel_off(path, res, unpackbits=True, min=-0.5, max=0.5):
    voxel_path = path + '/voxelization_{}.npy'.format( res)
    off_path = path + '/voxelization_{}.off'.format( res)

    if unpackbits:
        occ = np.unpackbits(np.load(voxel_path))
        voxels = np.reshape(occ, (res,)*3)
    else:
        voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,)*3)

    loc = ((min+max)/2, )*3
    scale = max - min

    VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
    print('Finished: {}'.format(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization to off'
    )
    parser.add_argument('-res', type=int)

    args = parser.parse_args()

    ROOT = 'shapenet/data'

    p = Pool(mp.cpu_count() // 2)
    # p = Pool(4)
    p.map(partial(create_voxel_off, res=args.res), glob.glob( ROOT + '/*/*/'))