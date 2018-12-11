from gluoncv import model_zoo, data, utils
import pickle
import os

dataset = './images'
model = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
dic = {}
for pasta in os.listdir(dataset):
    if(pasta[0] != '.'):
        for imgs in os.listdir(dataset + '/' + pasta):
            image_path = dataset + '/' + pasta +'/' + imgs
            x, img = data.transforms.presets.yolo.load_test(image_path, short=512)
            class_IDs, scores, bounding_boxs = model(x)
            scores = scores.asnumpy()
            class_IDs = class_IDs.asnumpy()
            bounding_boxs = bounding_boxs.asnumpy()
            for j in range(len(class_IDs[0])):
                if(scores[0][j] > 0.3):
                    index = int(class_IDs[0][j][0])
                    if(model.classes[index] not in dic):
                        dic[model.classes[index]] = []
                    if(image_path not in dic[MODEL.classes[index]]):
                        dic[model.classes[index]].append(image_path)


PATH = './results/train.p'
pickle.dump(dic, open(PATH, "wb"))
