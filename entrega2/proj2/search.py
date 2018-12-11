import sys
import pickle
from pprint import pprint


dic = pickle.load(open('./results/train.p', 'rb'))
input_label = sys.argv[1]
possible_images = dic[input_label]
print('Voce procurou por:', input_label)
print('Possiveis imagens:')
pprint(possible_images)
