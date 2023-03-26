import os
import glob
from tqdm import tqdm
import pickle
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        # dir = os.path.dirname(__file__)
        self.data_dir = './images/'
        self.label_num_dic = {0 : 'kirimi', 1 : 'other'}
        self.img_size = 50
        self.pkl_file = "./dataset.pkl"
        

    def create_dataset(self):
        _dataset = {}
        _dataset_x = []
        _dataset_t = []
        img_size = (self.img_size, self.img_size)
        img_path = glob.glob(self.data_dir + '**/*.png')
        for path in img_path:
            label = int(os.path.dirname(path).split('/')[-1])
            try:
                    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # load images
                    img_resize_array = cv2.resize(img_array, img_size)  # resize images
                    _dataset_x.append(img_resize_array)
                    _dataset_t.append(label)
            except Exception as e:
                pass
        _dataset['train_image'] = np.array(_dataset_x)
        _dataset['train_label'] = np.array(_dataset_t)

        return _dataset

    def init_dataset(self):
        print('Creating dataset ...', end='')
        self.dataset = self.create_dataset()

        with open(self.pkl_file, 'wb') as f:
            pickle.dump(self.dataset, f, -1)
        print(' -> Done!')

    def load_dataset(self, normalize=True, flatten=True, one_hot_label=False):
        if not os.path.exists(self.pkl_file):
            self.init_dataset()

        print('Now loading dataset ...', end='')
        with open(self.pkl_file, 'rb') as f:
            self.dataset = pickle.load(f)
        print(' -> Done!')

        x_train = self.dataset['train_image']
        t_train = self.dataset['train_label']

        # print(x_train.shape) # (7,50,50)
        # print(t_train)       # [0 0 0 0 0 1 1]

        if normalize:
                # print(x_train[0][0][0])
                x_train = x_train.astype(np.float32)
                x_train /= 255.0
                # print(x_train[0][0][0])

        # if one_hot_label:
        #     dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
            # dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

        # if not flatten:
        #     for key in ('train_img', 'test_img'):
        #         dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
        if flatten:
            # print(x_train.shape)
            arr_size = self.img_size ** 2
            x_train = x_train.reshape(-1, arr_size)
            # print(x_train.shape)

        #dataset # (7, 25, 25)
        self.x_train = x_train # (7, 2500)
        self.t_train = t_train

        # return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

    def check_dataset(self):
        # データセットの確認
        idx_lis = np.random.randint(0, len(self.dataset['train_image']), 4)
        for idx,i in enumerate(idx_lis):
            label = self.label_num_dic[self.dataset['train_label'][i]]
            print("training data label : ", label)
            plt.subplot(2, 2, idx+1)
            plt.axis('off')
            plt.title(label=label)
            plt.imshow(self.dataset['train_image'][i], cmap='gray')

        plt.show()

if __name__ == '__main__':
    cls = Dataset()
    cls.load_dataset()
    cls.check_dataset()
