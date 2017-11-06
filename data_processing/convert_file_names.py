import os,sys

def convert_file_names(dir_path, elevation_angle):
	failures = []
	for path, subdirs, files in os.walk(dir_path):
		for name in files:
			if name.endswith('.png'):
				if len(name.rsplit('-',1)) != 2:
					failures.append(os.path.join(path, name))
					continue
				model_id, img_id = name.rsplit('-', 1)
				azimuth_angle, extension = os.path.splitext(img_id)
				new_name = model_id + '_' + azimuth_angle + '_' + str(elevation_angle) + extension

				current_path = os.path.join(path, name)
				new_path = os.path.join(path, new_name)
				os.rename(current_path, new_path)

	print(failures)

if __name__=='__main__':
	convert_file_names(sys.argv[1], sys.argv[2])
	# python convert_file_names.py '/home/chinmay/CODE/shapenet-viewer/rendered_data/car' 30
	# python convert_file_names.py '/home/chinmay/CODE/deep_learning/shapenet_datasets' 0