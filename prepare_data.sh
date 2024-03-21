python -m data_processing.voxelize -res 128
python -m data_processing.boundary_sampling -sigma 0.1
python -m data_processing.boundary_sampling -sigma 0.01
python -m data_processing.create_voxel_off -res 128