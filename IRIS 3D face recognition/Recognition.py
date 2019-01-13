from __future__ import print_function
import caffe
import numpy as np
import scipy.misc
from os import listdir
from os.path import isfile, join
import time
from sklearn.metrics.pairwise import cosine_similarity
import sys


def getFeatures(net, adata):
    sFeatures = []
    for i in range(0,len(adata)):
        img = adata[i]
        net.blobs['data'].data[...] = img
        out = net.forward()

        caffe_fc7 = net.blobs['fc7'].data[0].copy()
        sTemp_layer_output = caffe_fc7
        sTemp_layer_output= sTemp_layer_output.reshape(1, 4096)

        sTemp_layer_output = np.sqrt(sTemp_layer_output)
        sFeatures.append(sTemp_layer_output)
    sFeatures = np.array(sFeatures)
    sFeatures = sFeatures.reshape(len(sFeatures), 4096)
    return sFeatures

def getIdentificationAccuracy(model, aGallery, aGalleryLabel,aProb, aProbLabel):
    i = 0
    sGalleryFeature = getFeatures(model, aGallery)

    sProbFeature = getFeatures(model, aProb)

    minindex = []
    similarities = []
    max_similarities = []
    similaritiesMatrix = []

    for i in range(0, sProbFeature.shape[0]):
        if i == 0:
          t0 = time.clock()
        temp_max = -10
        temp_index = 0;
        temp_max = -1000

        measure = []

        for j in range(0, sGalleryFeature.shape[0]):
            temp_similarity = cosine_similarity(sProbFeature[i,:].reshape(1,-1), sGalleryFeature[j,:].reshape(1,-1))
            if temp_max < temp_similarity:
                temp_max = temp_similarity
                temp_index = aGalleryLabel[j]

            if aGalleryLabel[j] == aProbLabel[i]:
                similarities.append(temp_similarity)

            measure.append(temp_similarity)

        max_similarities.append(temp_max)
        minindex.append(temp_index)
        similaritiesMatrix.append(np.squeeze(np.array(measure)))


    minindex = np.array(minindex)
    results = (minindex == aProbLabel)
    accuracy = np.count_nonzero(results)/float(results.shape[0])

    return accuracy, results, minindex, max_similarities, similarities, np.array(similaritiesMatrix)

def rankAccuracy(sProbLabel, similaritiesMatrix , k):

    RankLabel = []
    for i in range(0,similaritiesMatrix.shape[0]):
        ind = np.argpartition(-(similaritiesMatrix[i,:]), k)[:k]
        if sProbLabel[i] in ind:
            RankLabel.append(True)
        else:
            RankLabel.append(False)

    RankLabel = np.array(RankLabel)
    return  np.count_nonzero(RankLabel)/float(similaritiesMatrix.shape[0])


def to_rgb1a(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
    return ret


def InitCnn():
    caffe.set_mode_cpu()
    net = caffe.Net("VGG_FACE_deploy.prototxt", "Weights.caffemodel", caffe.TEST)

    print('The CNN was successfully initiated')
    return net

def ParseDir(path):

    avg = np.array([37, 37, 37])
    s_target = '.npy'

    x__path = []
    y__path = []
    file_names = []

    dirs = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith(s_target))]
    n_id = len(dirs)
    for i, n in enumerate(dirs):
        y_temp = np.zeros(n_id)
        y_temp[i] = 1
        x_temp = np.load(path + '/' + n)
        x_temp = to_rgb1a(x_temp)
        image = scipy.misc.imresize(x_temp.astype('float32'), [224, 224])
        image = image - avg
        image = image.transpose((2, 0, 1))

        x__path.append(image)
        y__path.append(y_temp)
        file_names.append(n)

    x__path = np.array(x__path)
    y__path = np.array(y__path)
    print(path, ' data shape: ', x__path.shape)

    return x__path, y__path, file_names

def Recognize(gallery_path, probe_path):
    net = InitCnn()

    x__gallery, y__gallery, file_names_gallery = ParseDir(gallery_path)

    x__probe, y__probe, file_names_probe = ParseDir(probe_path)

    s_gallery_label = np.where(y__gallery == 1)
    s_gallery_label = s_gallery_label[1]

    s_probe_label = np.where(y__probe == 1)
    s_probe_label = s_probe_label[1]

    [results, label, min, max_similarities, similarities, similaritiesMatrix] = getIdentificationAccuracy(net,
                                                                                                          x__gallery,
                                                                                                          s_gallery_label,
                                                                                                          x__probe,
                                                                                                          s_probe_label)

    print("rank-1 acc: ", results)

    for i in range(0, len(file_names_probe)):
        try:
            print(file_names_probe[i],
                  'Probe Label:', s_gallery_label[i],
                  'Matched Label:', file_names_gallery[min[i]],
                  'max similarity: ', max_similarities[i],
                  'ref similarity: ', similarities[i],
                  'matches: ', similaritiesMatrix[i])
            print('\n')
        except:
            print('err', i)


# Application main
if __name__ == '__main__':
    sGalPath = 'E:/BosphorusDB/ply/gallery'
    sProbPath = 'E:/BosphorusDB/ply/probe'
    Recognize(sGalPath, sProbPath)
