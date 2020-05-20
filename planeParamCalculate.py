import numpy as np
import os

root = "/home/ogam/Documents/mimari-proje/class1/flir/outside_out_pmvs/reconstruction_sequential/PMVS_the/"
os.chdir(root)

with open('planes/calculated_planes.txt', 'rb') as the_file:
    normals = np.loadtxt(the_file, delimiter=' ')

plane1, d1 = normals[0,:3], normals[0,3]
x0_1 = d1*plane1/np.dot(plane1.T, plane1)

eq1 = normals[0,:]/-normals[0,2]

