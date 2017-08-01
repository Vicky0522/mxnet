import mxnet as mx
import numpy as np
import math
import struct

# read MNIST data
# training set: 60000 examples 28x28
# test set: 10000 examples 28x28

train_data_path = './Datasets/MNIST/train-images-idx3-ubyte'
train_label_path = './Datasets/MNIST/train-labels-idx1-ubyte'
test_data_path = './Datasets/MNIST/t10k-images-idx3-ubyte'
test_label_path = './Datasets/MNIST/t10k-labels-idx1-ubyte'

with open(train_data_path,'rb') as f:
    magic, train_data_size, height, width = struct.unpack('>IIII', f.read(16))
    train_data = np.fromfile(f,dtype=np.uint8).reshape((train_data_size,1,28,28))
    
with open(train_label_path,'rb') as f:
    f.seek(8)
    train_label = np.fromfile(f,dtype=np.uint8)
    
with open(test_data_path,'rb') as f:
    magic, test_data_size, height, width = struct.unpack('>IIII', f.read(16))
    test_data = np.fromfile(f,dtype=np.uint8).reshape((test_data_size,1,28,28))

with open(test_label_path,'rb') as f:
    f.seek(8)
    test_label = np.fromfile(f,dtype=np.uint8)

#normalization
train_data = train_data.astype(np.float32, copy=False)
test_data = test_data.astype(np.float32, copy=False)
mean = train_data.mean()
train_data -= mean
std = train_data.std()
train_data /= std
test_data = (test_data-mean)/std


MNIST_train_dataIter = mx.io.NDArrayIter(data=train_data, 
                                         label=train_label, 
                                         batch_size=100, 
                                         shuffle=True)
MNIST_test_dataIter = mx.io.NDArrayIter(data=test_data, 
                                        label=test_label, 
                                        batch_size=100)

# mnist = mx.test_utils.get_mnist()
# batch_size = 100
# MNIST_train_dataIter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
# MNIST_test_dataIter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


#construct Lenet


data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc layer
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet5 = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
#mx.viz.plot_network(lenet5)



# train model
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

lenet5_model = mx.mod.Module(symbol=lenet5, context=mx.gpu())
lenet5_model.fit(MNIST_train_dataIter, 
                 eval_data=MNIST_test_dataIter,
                 optimizer='sgd',
                 optimizer_params={'learning_rate':0.1},
                 eval_metric='acc',
                 batch_end_callback = mx.callback.Speedometer(100, 100),
                 num_epoch=10)
      
