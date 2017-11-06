# Multi-Viewpoint-Image-generation
Given input view points, create 2D object images in a different viewpoint.
This is WIP.

## Run command

Download Shapenet dataset

Run command to rename shapenet files
python data_processing/convert_file_names.py <data_set_path> <elevation_angle>

Save dataset path in run_model.py

python run_model.py config.json
