import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import argparse
import torch
import numpy as np
import signal

def main():
    # python train.py -posed -dist 0.5 0.5 -std_dev 0.15 0.05 -res 32 -batch_size 40 -m
    parser = argparse.ArgumentParser(
        description='Run Model'
    )


    parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
    parser.add_argument('-voxels', dest='pointcloud', action='store_false')
    parser.set_defaults(pointcloud=False)
    parser.add_argument('-pc_samples' , default=3000, type=int)
    parser.add_argument('-dist','--sample_distribution', default=[0.5, 0.5], nargs='+', type=float)
    parser.add_argument('-std_dev','--sample_sigmas',default=[0.15,0.015], nargs='+', type=float)
    parser.add_argument('-batch_size' , default=30, type=int)
    parser.add_argument('-res' , default=32, type=int)
    parser.add_argument('-m','--model' , default='LocNet', type=str)
    parser.add_argument('-o','--optimizer' , default='Adam', type=str)
    parser.add_argument('-w','--workers' , default=12, type=int)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]


    if args.model ==  'ShapeNet32Vox':
        net = model.ShapeNet32Vox()

    if args.model ==  'ShapeNet128Vox':
        net = model.ShapeNet128Vox()

    if args.model == 'ShapeNetPoints':
        net = model.ShapeNetPoints()

    if args.model == 'SVR':
        net = model.SVR()

    split = np.load('shapenet/split.npz')

    train_dataset = voxelized_data.VoxelizedDataset(split['train'], voxelized_pointcloud= args.pointcloud, pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=args.workers)

    val_dataset = voxelized_data.VoxelizedDataset(split['val'], voxelized_pointcloud= args.pointcloud , pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=args.workers)



    exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(  'PC' + str(args.pc_samples) if args.pointcloud else 'Voxels',
                                        ''.join(str(e)+'_' for e in args.sample_distribution),
                                        ''.join(str(e) +'_'for e in args.sample_sigmas),
                                                                    args.res,args.model)

    def signal_handler(sig, frame):
        print('Terminating')
        trainer.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    trainer = training.Trainer(net,torch.device("cuda"),train_dataset, val_dataset,exp_name, optimizer=args.optimizer)
    trainer.train_model(1500)

if __name__ == '__main__':
    main()