#predict
from two_layer_net import TwoLayerNet
from dataset import Dataset
import glob
import cv2
import pickle
import numpy as np

test_img_dir = './images/'

test_imgs = glob.glob(test_img_dir+'*.png')

# cleate network
nw = TwoLayerNet(2500,1,2)

# read paramaters from pickle file
with open('model.pkl','rb') as f:
    nw.params = pickle.load(f)

ds = Dataset()
img_size = (50, 50)

for path in test_imgs:
    # preproccess
    # path = './images/sub-kirimichan-samesenpai.png'
    x_test = ds.preproccess(path, img_size)
    # nomalize
    x_test = x_test.astype(np.float32)
    x_test /= 255.0
    # flatten
    x_test = x_test.reshape(-1, 50**2)

    y = nw.predict(x_test)

    print(y, y.argmax())
    print(path, '-> ' + ds.label_num_dic[y.argmax()])
