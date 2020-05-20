"""
sample_rate: the propotion of clients selected in every communication rounds
number_client: the number of client 
model_type: model to be trained
            CNN_sketch: CNN with sketch
            CNN: standard CNN 
            NN: multilayer perceptron
            MLP_SketchLinear: multilayer perceptron with Sketch

datatype: cifar/ mnist
dim_in: input dimension, only for mnist data
dim_out: dimension of output class, only for mnist data
p: parameter for random hashing in Sketch
round: communication rounds between Server and Clients
local_epochs: number of epochs for each client in one communication
local_batch_size: batch size for each client during local training 
learningrate_client: learning rate for each client during local training 
test_batch_size: batch size for test data
verbose: verbose setting for test
target: target accuracy for training to stop
gpu: 1 use gpu when avaliable
     -1 use cpu

"""



import torch

class Args:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.sample_rate = 0.01
        self.number_client = 100
        self.model_type = 'CNN_sketch'
        self.datatype = 'cifar'
        self.dim_in = 784
        self.dim_out = 10
        self.p = 2
        self.round = 500
        self.local_epochs = 5
        self.local_batch_size = 50
        self.learningrate_client = 0.001
        self.test_batch_size = 1000
        self.verbose = 1
        self.target = 60
        self.gpu = 1
