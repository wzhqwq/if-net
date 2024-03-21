from functools import partial
import trimesh
import numpy as np
from . import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback
from sklearn.neighbors import KDTree

# from matplotlib import pyplot as plt

ROOT = 'shapenet/data'

gingival_sample_ratio = 0.7
bottom_sample_ratio = 0.05
teeth_sample_ratio = 0.25

blur_sigma = 0.01
gingival_weight = 1
bottom_weight = 1
teeth_min_weight = 0.1

def findContours(faces: np.ndarray):
    # find contour using face adjacency
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    return edges[counts == 1]

def boundary_sampling(path, args, sample_num = 100000):
    try:

        if os.path.exists(path +'/boundary_{}_samples.npz'.format(args.sigma)):
            return

        full_path = path + '/isosurf_scaled.off'
        gingival_path = path + '/gingival_scaled.off'
        bottom_path = path + '/bottom_scaled.off'
        teeth_path = path + '/teeth_scaled.off'
        out_file = path +'/boundary_{}_samples.npz'.format(args.sigma)

        mesh = trimesh.load(full_path)
        gingival = trimesh.load(gingival_path)
        gingival_points = gingival.sample(int(sample_num * gingival_sample_ratio))
        bottom = trimesh.load(bottom_path)
        bottom_points = bottom.sample(int(sample_num * bottom_sample_ratio))
        teeth = trimesh.load(teeth_path)
        teeth_points = teeth.sample(int(sample_num * teeth_sample_ratio))
        points = np.concatenate([gingival_points, bottom_points, teeth_points])
        sample_num = len(points)

        teeth_trim_indices = np.unique(findContours(teeth.faces))

        tree = KDTree(mesh.vertices)
        teeth_trim_mask = np.zeros(len(mesh.vertices), dtype=bool)
        indices = tree.query_radius(teeth.vertices[teeth_trim_indices], r=0.02)
        teeth_trim_mask[np.unique(np.concatenate(indices))] = True
        # do gaussian blur on teeth_points
        indices, distances = tree.query_radius(teeth_points, r=0.04, return_distance=True)
        teeth_weights = []
        for i in range(len(teeth_points)):
            neighbor_weights = np.exp(-distances[i] ** 2 / (2 * args.sigma ** 2))
            neighbor_weights /= np.sum(neighbor_weights)
            neighbor_weights = neighbor_weights[teeth_trim_mask[indices[i]]]
            teeth_weights.append(np.sum(neighbor_weights))
        teeth_weights = np.clip(np.array(teeth_weights), teeth_min_weight, 1.0)
        weights = np.concatenate([
            np.full(len(gingival_points), gingival_weight),
            np.full(len(bottom_points), bottom_weight),
            teeth_weights
        ])
        # indices, distances = tree.query_radius(points, r=0.02, return_distance=True)
        # weights = []
        # for i in range(len(points)):
        #     neighbor_weights = np.exp(-distances[i] ** 2 / (2 * blur_sigma ** 2))
        #     neighbor_weights /= np.sum(neighbor_weights)
        #     neighbor_weights = neighbor_weights[teeth_trim_mask[indices[i]]]
        #     weights.append(np.sum(neighbor_weights))
        # weights = np.clip(np.array(weights), teeth_min_weight, 1.0)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        # presentation = trimesh.Scene()
        # presentation.add_geometry(mesh)
        # brightnesses = np.array(weights * 255, dtype=np.uint8)
        # colors = np.zeros((len(boundary_points), 4), dtype=np.uint8)
        # # colors[occupancies > 0, 1] = brightnesses[occupancies > 0]
        # # colors[occupancies <= 0, 0] = brightnesses[occupancies <= 0]
        # colors[:, 0] = brightnesses
        # colors[:, 3] = 255
        # # pct = trimesh.PointCloud(boundary_points, colors=colors)
        # pct = trimesh.PointCloud(points, colors=colors)
        # # pct.show()
        # presentation.add_geometry(pct)
        # pct.export('rest.ply')
        # presentation.show()

        # transformed_points = boundary_points
        # transformed_points = transformed_points + np.array([0, 0, 0.1])
        # in_points = transformed_points[occupancies > 0]
        # in_points = in_points[np.abs(in_points[:, 2]) < 0.005]
        # out_points = transformed_points[occupancies <= 0]
        # out_points = out_points[np.abs(out_points[:, 2]) < 0.005]
        # plt.scatter(in_points[:, 0], in_points[:, 1], c='b')
        # plt.scatter(out_points[:, 0], out_points[:, 1], c='r')
        # plt.show()

        np.savez(out_file, points=boundary_points, occupancies=occupancies, grid_coords=grid_coords, weights=weights)
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
    # boundary_sampling(glob.glob( ROOT + '/*/*/')[0], args=args)
