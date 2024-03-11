from functools import partial
import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback

# from matplotlib import pyplot as plt

ROOT = 'shapenet/data'


def boundary_sampling(path, args, sample_num = 100000):
    try:

        if os.path.exists(path +'/boundary_{}_samples.npz'.format(args.sigma)):
            return

        off_path = path + '/isosurf_scaled.off'
        out_file = path +'/boundary_{}_samples.npz'.format(args.sigma)

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        # transformed_points = boundary_points
        # transformed_points = transformed_points + np.array([0, 0, 0.1])
        # in_points = transformed_points[occupancies > 0]
        # in_points = in_points[np.abs(in_points[:, 2]) < 0.005]
        # out_points = transformed_points[occupancies <= 0]
        # out_points = out_points[np.abs(out_points[:, 2]) < 0.005]
        # plt.scatter(in_points[:, 0], in_points[:, 1], c='b')
        # plt.scatter(out_points[:, 0], out_points[:, 1], c='r')
        # plt.show()

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float)

    args = parser.parse_args()

    p = Pool(mp.cpu_count())
    p.map(partial(boundary_sampling, args=args), glob.glob( ROOT + '/*/*/'))
