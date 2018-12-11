import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import namedtuple
import os, sys
from random import shuffle
from sklearn.cluster import KMeans
from scipy.stats import chisquare
from random import shuffle
import pickle

orb = cv2.ORB_create()

def return_descriptor(img):
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp,des

def computa_descritores(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)    
    return des

def le_descritores_imagens(pastas, max_items = 5):
    list_Names  = []
    list_Matrix = []
    list_Imgs   = []
    Tuple = namedtuple('Tupla', 'listNames, matrix ')
    
    for pasta in pastas:
        dir_Name = "101_ObjectCategories/" + pasta + '/'
        img_Name = os.listdir(dir_Name)
        #shuffle(img_Name)
            
        for i in range(max_items):
            if(img_Name[i][0]=="."):
                img_Name[i]=img_Name[i][2:]
            name = dir_Name + img_Name[i]
            list_Imgs.append(cv2.imread(name))
            list_Names.append(name)
            
    for img in list_Imgs:
        list_Matrix.append(computa_descritores(img))
    
    tup = Tuple(list_Names, np.concatenate(list_Matrix)) 
    return tup

def cria_vocabulario(descritores, sz = 300):                

    kmeans = KMeans(n_clusters = sz, random_state=0).fit(descritores)
    tup = [kmeans.cluster_centers_ , kmeans]

    return tup

def representa_histograma(img, vocab):
    desc = computa_descritores(img)
    
    dist = vocab[1].predict(desc)
    
    list_freq = [0] * 300
    
    for i in dist:
        list_freq[i] += 1
   
    return list_freq

def similarity(hist,list_Hist):    
    dist = []
    index=0
    for i in list_Hist:
        chi_squared = compara_histograma(hist, i)
        dist.append((index, chi_squared))
        print(index,chi_squared)
        index+=1

    distances = sorted(dist, key=lambda x: x[1], reverse=False)
    index=[]
    for match in distances[:5]:
        index.append(match[0])
    return index

#funcao feita pelo hugo, chisquate do scipy.stats nao funcionou
def compara_histograma(hist, hist_vocab):
    hist1  = np.array(hist.copy()) + 1
    hist2  = np.array(hist_vocab.copy()) + 1 
    dist = 0
    for i in range(0, len(hist1)):
        if(hist1[i] == 0):
            temp = 1
        else:
            temp = ((hist1[i] - hist2[i]) ** 2) / hist1[i]
    dist += temp
    return dist

def train(pasta, max_items=15):
    descritores = le_descritores_imagens(pasta, max_items)
    vocab = cria_vocabulario(descritores[1])
    list_Path  = []
    list_Hist = []
    list_Imgs   = []
    for p in pasta:
        dir_Name = "101_ObjectCategories/" + p + '/'
        img_Name = os.listdir(dir_Name)
            
        for i in range(max_items):
            if(img_Name[i][0]=="."):
                img_Name[i]=img_Name[i][2:]
            name = dir_Name + img_Name[i]
            list_Imgs.append(cv2.imread(name))
            list_Path.append(name)
            
    for img in list_Imgs:
        list_Hist.append(representa_histograma(img,vocab))
    pickle.dump(vocab, open("vocab.p", "wb"))
    pickle.dump(list_Hist, open("list_Hist.p", "wb"))
    pickle.dump(list_Path, open("list_Path.p", "wb"))


def main():
    pastas=["Faces", "garfield", "wheelchair", "elephant"]
    train(pastas)
    

if __name__ == '__main__':
    main()
