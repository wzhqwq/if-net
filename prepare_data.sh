python data_processing/voxelize.py -res 128
python -m data_processing.boundary_sampling.py -sigma 0.1
python -m data_processing.boundary_sampling.py -sigma 0.01
python data_processing/create_voxel_off.py -res 128