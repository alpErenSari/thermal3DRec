import numpy as np
import pandas as pd
import os, glob
import cv2

root = "/home/ogam/Documents/mimari-proje/class1/flir/outside_out_pmvs/reconstruction_sequential/PMVS_the/"
os.chdir(root)

projectionMatrices = glob.glob('txt/*')
projectionMatrices.sort(reverse=False)

rgb_intrinsic = np.loadtxt('parameters/electroptic_camera.txt', delimiter=',')
thermal_intrinsic = np.loadtxt('parameters/thermal_camera.txt', delimiter=',')
rot_diff_mat = np.loadtxt('parameters/rotation_difference.txt', delimiter=',')
trans_diff_ = np.loadtxt('parameters/translation_difference_2.txt', delimiter=',')

#projection = pd.read_csv(projectionMatrices[0], delim_whitespace=True)
#pr_ = np.loadtxt(projectionMatrices[0])

with open(projectionMatrices[0], 'r') as f:
    x = f.readlines()
    
x.remove(x[0])
x = list(map(lambda a: a.split(), x))
x_ = np.float64(x)

thermalImages = glob.glob('visualize_th/*.jpg')
thermalImages.sort(reverse=False)
img_thermal = cv2.imread(thermalImages[0])
img_thermal_rgb = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2RGB)
#multiply the projection matrix with the inverse intrinsic matrix 
# to get rotation and translation
rot_trans = np.dot(np.linalg.pinv(rgb_intrinsic), x_)
# split the matrix into rotation and translation
rot_, trans_ = rot_trans[:,0:3], rot_trans[:,3]
rot_the = np.dot(rot_diff_mat, rot_)
trans_the = np.dot(rot_diff_mat, trans_) + trans_diff_
# put together new rotation and translation
rot_trans_the = np.concatenate((rot_the, np.reshape(trans_the, (-1,1))), axis=1)

# projection matrix of thermal
project_the = np.dot(thermal_intrinsic, rot_trans)

with open('models/pmvs_options.txt.ply', 'r') as f:
    ply = f.readlines()

ply = list(map(lambda a: a.split(), ply))
# we are taking only the numpber part of ply file
ply_ = np.float64(ply[13:])
# just take 3D point coordinates and append 1 to make them homogenous
coordinates = ply_[:,0:4].T
coordinates[3, :] = 1
# project the points and 
coordinates_ = np.dot(project_the, coordinates)
coord_norm = coordinates_/coordinates_[2,:]

mask = (coord_norm[0,:]>0)&(coord_norm[1,:]>0)&(coord_norm[0,:]<320)&(coord_norm[1,:]<240)
items = np.argwhere(mask)

coord_norm = np.int32(coord_norm)
ply_[items, -3:] = img_thermal_rgb[coord_norm[1,items], coord_norm[0,items], :]


with open('models/thermal_point.ply', 'w') as the_file:
    for item in ply[:13]:
        the_file.write(' '.join(item) + '\n')
    np.savetxt(the_file, ply_, delimiter=' ', fmt='%6f %6f %6f %6f %6f %6f %d %d %d')
    #np.savetxt(the_file, ply_[:,-3:], delimiter=' ', fmt='%d')

coordinates_rgb = np.dot(x_, coordinates)
coord_norm_rgb = coordinates_rgb/coordinates_rgb[2,:]

mask_rgb = (coord_norm_rgb[0,:]>0)&(coord_norm_rgb[1,:]>0)&(coord_norm_rgb[0,:]<1536)&(coord_norm_rgb[1,:]<2048)
items_rgb = np.argwhere(mask_rgb)



