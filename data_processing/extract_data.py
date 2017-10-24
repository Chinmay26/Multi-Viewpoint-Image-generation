#script to contruct configuration file by extracting object_id from shapenet metadata
#the config file generated will be used to render shapenet objects
import sys, csv

def construct_conf(config_fpath, obj_path, dst_path):
	viewer_commands = []
	model_object_ids = []
	with open(obj_path) as f:
		reader = csv.DictReader(f, delimiter=',')
		for row in reader:
			model_object_ids.append(row['fullId'].strip())

	#construct viewer.commands
	for id in model_object_ids:
		viewer_commands.append("load model {model_id}".format(model_id=id))
		viewer_commands.append("save model screenshots")


	#copy base config file to dst_path
	with open(config_fpath) as f1:
		with open(dst_path, 'w') as f2:
			for line in f1:
				f2.write(line)

	with open(dst_path, 'a') as f:
		f.write("\n\nviewer.commands = [\n")
		for i,k in enumerate(viewer_commands):
			f.write('"')
			f.write(k)
			f.write('"')
			if i < len(viewer_commands) - 1:
				f.write(',')
			f.write('\n')
		f.write("]")

if __name__=='__main__':
	# python extract_data.py base_config.conf chair_shapenet_metadata.csv chair_config.conf
	config_fpath = sys.argv[1] # base_config file './base_config.conf'
	obj_path= sys.argv[2] # metadata_file from shapenet'./chair_shapenet_metadata.csv'
	dst_path= sys.argv[3] # final_confif file 'chair_config.conf'
	construct_conf(config_fpath, obj_path, dst_path)

	
