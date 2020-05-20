import torch
from torch import nn
import torch.nn.functional as F
from model.SketchLinear import SketchLinear
from model.SketchConv import SketchConv



# Multilayer perceptron
# Args:
    #    dim_in: input dimension
    #    dim_out: output dimension
# Return:
    #    log probabilities of the classes
class NN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = [200, 200]

        self.input_layer = nn.Linear(self.dim_in, self.hidden[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden[i], self.hidden[i + 1])
                                            for i in range(0, len(self.hidden) - 1)])

        self.output_layer = nn.Linear(self.hidden[-1], self.dim_out)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        output = self.input_layer(x)
        output = self.activation(output)
        for hidden_layers in self.hidden_layers:
            output = hidden_layers(output)
        output = self.activation(output)
        output = self.output_layer(output)
        output = self.log_softmax(output)
        return output

# Multilayer perceptron with sketch
# Args:
    #    dim_in: input dimension
    #    dim_out: output dimension
    #    p: parameter for random hashing in Sketch
# Return:
    #    log probabilities of the classes
class MLP_SketchLinear(nn.Module):
    def __init__(self, dim_in, dim_out, p):
        super(MLP_SketchLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.p = p
        self.hidden = [200, 200]

        self.input_layer = SketchLinear(dim_in, self.hidden[0], p)
        self.activation = nn.ReLU()
        self.hidden_layers = nn.ModuleList([SketchLinear(self.hidden[i], self.hidden[i + 1], p)
                                            for i in range(0, len(self.hidden) - 1)])
        self.output_layer = SketchLinear(self.hidden[-1], dim_out, p)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        output = self.input_layer(x)
        output = self.activation(output)
        for hidden_layers in self.hidden_layers:
            output = hidden_layers(output)
        output = self.activation(output)
        output = self.output_layer(output)
        output = self.log_softmax(output)
        return self.log_softmax(output)

# CNN for mnist dataset
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# CNN with Sketch for mnist dataset
class CNNMnist_Sketch(nn.Module):
    def __init__(self, q):
        super(CNNMnist_Sketch, self).__init__()
        # self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.conv1 = SketchConv(1, 32, kernel_size=5, q=q)
        self.conv2 = SketchConv(32, 64, kernel_size=5, q=q)
        self.fc1 = SketchLinear(1024, 512, q=q)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# CNN for cifar dataset
class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# CNN with Sketch for cifar dataset
class CNNCifar_Sketch(nn.Module):
    def __init__(self, q):
        super(CNNCifar_Sketch, self).__init__()
        self.q = q
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = SketchConv(32, 64, kernel_size=5, q=self.q)
        self.conv3 = SketchConv(64, 128, kernel_size=5, q=self.q)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = SketchLinear(2048, 200, q=self.q)
        self.fc2 = nn.Linear(200, 10)
        self.fc2 = SketchLinear(200, 10, q=q)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


