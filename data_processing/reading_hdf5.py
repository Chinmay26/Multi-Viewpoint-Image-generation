# -*- coding: utf-8 -*-


import numpy as np
import h5py
import os
from scipy import misc
import pickle

class Shapenet(object):
    def __init__(self, folder_path):
        self.path = folder_path

    
#    def pickle_data(self,images):
#        print("reading")
#        with open('parrot.pkl', 'wb') as f:
#            pickle.dump(images, f, 2)
        
    def store_to_hdf5(self,images):
        n = len(images)
        f = h5py.File('data.hdf5','w')
        shape_dataset = np.array(images[0]).shape
        sh = (n,shape_dataset[0],shape_dataset[1],shape_dataset[2],3)
        dst = f.create_dataset("myimages", shape=sh,
                           dtype=np.uint8, compression='gzip')
        print(np.array(images[0]).shape)
        for each in range(n):
            dst[each] = np.array(images[each])
        print("Done storing")
        
    def read_from_hdf5(self,hdf5_path):
        data = []
        f = h5py.File(hdf5_path,'r')
        key = (list(f.keys())[0])
        data = list(f[key])
        print("Rows of data added:",len(data))
        
    '''
    Load images from the directory path.
    Assumes images are of the format imgid_azimuthangle_elevationangle. Ex: 1we2wd_0_0
    '''
    def get_dataset(self, train_split=0.8, azimuth_angles=range(0,360,10), elevation_angle=[0]):
        images = []
        prev_azimuth_angle = 0
        print("reading dataset:")    
        #Read all the images

        for subdir, dirs, files in os.walk(self.path):
            pose_image_list = []
            for fname in files:
                if fname.endswith('.png'):
                    fp = os.path.join(self.path, subdir, fname)
                    img_data = misc.imread(fp).astype(np.int32)                    
                    name,extension = os.path.splitext(fname)
                    fpart, elevation_angle = name.rsplit('_',1)
                    img_id, azimuth_angle = fpart.rsplit('_', 1)
                    
                    if prev_azimuth_angle == int(azimuth_angle) or  prev_azimuth_angle==0:
                        pose_image_list.append(img_data)
                    elif prev_azimuth_angle != int(azimuth_angle) and pose_image_list:
                        images.append(pose_image_list)
                        pose_image_list = []
                        pose_image_list.append(img_data)
                        
                    prev_azimuth_angle = int(azimuth_angle)
            if pose_image_list:
                images.append(pose_image_list)
        return images    
                
                

                    

         
if __name__=='__main__':

    #Change path    
    dpath = "C:/Users/nitis/OneDrive/Documents/Deep Learning/Project/datasets/mug/models/3dw"
 #   dpath = "C:/Users/nitis/OneDrive/Documents/Deep Learning/Project/datasets/test/car_shapenet_dataset/3dw"
    
    s = Shapenet(dpath)
    trainData = s.get_dataset()
#    s.pickle_data(trainData)
    s.store_to_hdf5(trainData)         
    
    '''
    Call this function only while reading data from hdf5 file
    Replace data.hdf5 with hdf5 file name  
    '''
    #s.read_from_hdf5('data.hdf5')
    print("The End!")
