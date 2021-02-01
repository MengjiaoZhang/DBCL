import torch
import numpy as np

random_seed = 6666
np.random.seed(random_seed)
torch.manual_seed(random_seed)

from model.Server import Server
from model.Client import Client

from utils import load_mnist, sample_iid, load_cifar
from conf import Args


args = Args()

if args.datatype == 'mnist':
    path = './data/mnist'
    train_data, test_data = load_mnist(path)
elif args.datatype == 'cifar':
    path = './data/cifar'
    train_data, test_data = load_cifar(path)


data_split = sample_iid(train_data, args.number_client)


clients = []
for i in range(args.number_client):
    client = Client(train_data, data_split[i], args)
    clients.append(client)

server = Server(clients, test_data, args)

server.init_paras()
# from torchsummary import summary
# summary(server.server_model, (1, 784))
# exit()
server.train()


