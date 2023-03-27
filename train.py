from two_layer_net import TwoLayerNet
from common.functions import *
from common.gradient import numerical_gradient
from dataset import Dataset
from tqdm import  tqdm
import pickle

# dataset
ds = Dataset()
ds.load_dataset()

x = ds.x_train
t = ds.t_train

# network
input_size = ds.img_size ** 2
hidden_size = 1
output_size = len(ds.label_num_dic)

nw = TwoLayerNet(input_size, hidden_size, output_size)

h = 1e-4
for j in tqdm(range(100)):
    grad = {}
    # print('j=',j)
    for key in ('W1', 'b1', 'W2', 'b2'):
        # print('key=',key)
        _grad = np.zeros_like(nw.params[key])
        it = np.nditer(nw.params[key], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            #print('idx=',idx)
            # for i in range(len(grad)):
            # grad[0][0]
            tmp = nw.params[key][idx]

            nw.params[key][idx] = tmp + h
            y = nw.predict(x)
            fxh1 = cross_entropy_error(y, t)

            nw.params[key][idx] = tmp - h
            y = nw.predict(x)
            fxh2 = cross_entropy_error(y, t)

            _grad[idx] = (fxh1 - fxh2) / (2 * h)
            nw.params[key][idx] = tmp
            it.iternext()   
        grad[key] = _grad
    for key in ('W1', 'b1', 'W2', 'b2'):
            nw.params[key] -= 1 * grad[key]
    
print(grad)

print('Creating model ...', end='')
pkl_file  = './model.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(nw.params, f, -1)
print(' -> Done!')