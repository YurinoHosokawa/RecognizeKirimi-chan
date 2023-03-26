from two_layer_net import TwoLayerNet
from common.functions import *
from common.gradient import numerical_gradient
from dataset import Dataset
from tqdm import  tqdm

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
    for key in ('W1', 'b1', 'W2', 'b2'):
        grad = np.zeros_like(nw.params[key])
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
        # for i in range(len(grad)):
            # grad[0][0]
            tmp = nw.params[key][idx]

            nw.params[key][idx] = tmp + h
            y = nw.predict(x)
            fxh1 = cross_entropy_error(y, t)

            nw.params[key][idx] = tmp - h
            y = nw.predict(x)
            fxh2 = cross_entropy_error(y, t)

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            nw.params['W1'][idx] = tmp
    for key in ('W1', 'b1', 'W2', 'b2'):
            nw.params[key] -= 1 * grad[key]
    
print(grad)