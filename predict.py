from two_layer_net import TwoLayerNet
from dataset import Dataset
import glob
import pickle
import numpy as np

# test image
test_img_dir = 'test_images/*'
test_img_lis = glob.glob(test_img_dir)

path = test_img_lis[0]
for path in test_img_lis:
    img_size = (50, 50)

    ds = Dataset()
    x_test = ds.preproccess(path, img_size)
    x_test = x_test.astype(np.float32)
    x_test /= 255.0
    arr_size = 50 ** 2
    x_test = x_test.reshape(-1, arr_size)

    # read paramaters
    nw = TwoLayerNet(2500, 1, 2)

    pkl_file = 'model0.pkl'
    with open(pkl_file, 'rb') as f:
        nw.params = pickle.load(f)

    y = nw.predict(x_test)
    print(path)
    print(y.argmax())
