#from __future__ import print_function

import numpy as np

from plyfile import PlyData, PlyElement
from XYProjection import XYProjectedImage
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from os import listdir
import os
from os.path import isfile, join
from pprint import pprint
#exit()


def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))


def distance(v1,v2):
    return sum([(x-y)**2 for (x,y) in zip(v1,v2)])**(0.5)


def convert_ply_to_npy(path, s_save_path):
    if not os.path.exists(s_save_path):
        os.mkdir(s_save_path)

    files = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith('.ply'))]

    for file_path in files:
        try:
            pprint(file_path)
            sPointCloud = PlyData.read(path + '/' + file_path)
            sPointCloud = sPointCloud['vertex'][:]
            sPointCloudVerts = [list(row)[:3] for row in sPointCloud]
            sPointCloudVerts = np.array(sPointCloudVerts)

            normalizationFactor = 100.0
            normalizedVerts_ = []

            meanX = np.mean(sPointCloudVerts[:, 0])
            meanY = np.mean(sPointCloudVerts[:, 1])
            meanZ = np.mean(sPointCloudVerts[:, 2])

            for v in sPointCloudVerts:
                v[0] = (v[0] - meanX) / normalizationFactor
                v[1] = (v[1] - meanY) / normalizationFactor
                v[2] = (v[2] - meanZ) / normalizationFactor
                normalizedVerts_.append(v)

            sFileName = s_save_path + '/' + file_path[:-4] + ".npy"
            normalizedVerts = np.array(normalizedVerts_)

            sConverted = XYProjectedImage(normalizedVerts)
            np.save(sFileName, sConverted)
        except:
            print 'bad face %s' % file_path


# Application main
if __name__ == '__main__':

    ## Define Path

    #Paths = ['./3DFace/Probe', './3DFace/Gallery']
    Paths = ['E:/BosphorusDB/ply/probe', 'E:/BosphorusDB/ply/gallery']


    for sDefaultPath in Paths:
        convert_ply_to_npy(sDefaultPath, sDefaultPath)

