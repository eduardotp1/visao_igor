import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import namedtuple
import os, sys
from random import shuffle
from sklearn.cluster import KMeans
from scipy.stats import chisquare
from random import shuffle
import train
import pickle

caminho=sys.argv[1]

with open('vocab.p', 'rb') as handle:
    vocab = pickle.load(handle)

with open('list_Hist.p', 'rb') as handle:
    list_Hist = pickle.load(handle)

with open('list_Path.p', 'rb') as handle:
    list_Path = pickle.load(handle)



img = cv2.imread(caminho)
descs = train.computa_descritores(img)
hist =  train.representa_histograma(img,vocab)
similar = train.similarity(hist,list_Hist)
print(similar)

images = []
for i in similar:
    images.append(cv2.imread(list_Path[i]))  
for i in range(0,5):
    cv2.imshow('image',images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()