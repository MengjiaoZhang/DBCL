import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.Network import NN, MLP_SketchLinear, CNNCifar, CNNMnist, CNNCifar_Sketch, CNNMnist_Sketch


class Server:
    # build a server
    # broadcast parameters to clients
    # aggregate parameters collected from clients
    # update model parameters
    # test the global model after each communication round

    
    def __init__(self, clients, test_data, args):
        self.args = args
        self.clients = clients
        self.test_data = test_data
        self.clients_data_numbers = np.array([client.size() for client in self.clients])
        self.working_client = None
        self.init_paras()

    def init_paras(self):

        if self.args.model_type == 'NN':
            self.server_model = NN(self.args.dim_in, self.args.dim_out).to(self.args.device)
        elif self.args.model_type == 'MLP_SketchLinear':
            self.server_model = MLP_SketchLinear(self.args.dim_in, self.args.dim_out, self.args.p).to(self.args.device)
        elif self.args.model_type == 'CNN' and self.args.datatype == 'mnist':
            self.server_model = CNNMnist().to(self.args.device)
        elif self.args.model_type == 'CNN' and self.args.datatype == 'cifar':
            self.server_model = CNNCifar().to(self.args.device)
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'cifar':
            self.server_model = CNNCifar_Sketch(self.args.p).to(self.args.device)
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'mnist':
            self.server_model = CNNMnist_Sketch(self.args.p).to(self.args.device)
        self.server_model.train()
        self.global_weights = copy.deepcopy(self.server_model.state_dict())

    # collect local gradients from the selected clients in each communicaiton rounds
    def get_grads(self):
        self.local_grads = [self.clients[client_id].send_grads() for client_id in self.working_client]

    # randomly select some clients for training in each communicaiton rounds
    # broadcast global gradients to all selected clients
    def broadcast(self):
        num_client = int(len(self.clients) * self.args.sample_rate)
        self.working_client = np.random.choice(len(self.clients), num_client, replace=False)
        for client_id in self.working_client:
            self.clients[client_id].get_paras(copy.deepcopy(self.global_weights))


    # compute the average of the collected gradients 
    def _average(self, x):
        total_data_number = sum(self.clients_data_numbers[self.working_client])
        x_avg = copy.deepcopy(x[0])
        for k in x_avg.keys():
            x_avg[k] *= int(self.clients_data_numbers[self.working_client[0]])
        for k in x_avg.keys():
            for i in range(1, len(x)):
                x_avg[k] += x[i][k] * int(self.clients_data_numbers[self.working_client[i]])
            x_avg[k] = torch.div(x_avg[k], int(total_data_number))
        return x_avg

    # server updates the model parameters using averaged gradients
    def update_paras(self):
        self.get_grads()
        g_avg = self.average_grads()
        for k in g_avg.keys():
            self.global_weights[k] = self.global_weights[k] + g_avg[k]
        self.server_model.load_state_dict(self.global_weights)

    def average_grads(self):
        return self._average(self.local_grads)

    # clients train the local models on local date then send the gradients to server
    # server aggregates local gradients and updates the global model then sends global parameters to clients
    # server tests the global model after each communication rounds
    def train(self):
        accs, losses = [], []
        round = self.args.round
        for i in range(round):
            print('server round', i)
            self.broadcast()
            for client_id in self.working_client:
                print('client', client_id)
                client = self.clients[client_id]
                train_loss, train_acc = client.train(i)
                print('client', client_id, ' -- ', 'train loss:', train_loss, 'train_acc:', train_acc)
            self.update_paras()
            acc_test, test_loss = self.test()
            accs.append(str(float(acc_test)) + '\n')
            losses.append(str(float(test_loss)) + '\n')
            if acc_test >= self.args.target or i == (round-1):
                open('data/results/accs_' + self.args.datatype + self.args.model_type + self.args.datatype + '_lr_' + str(self.args.sample_rate)
                     + 'target_acc_' + str(self.args.target), 'w').writelines(accs)
                open('data/results/losses_' + self.args.datatype + self.args.model_type + self.args.datatype + '_lr_' + str(self.args.sample_rate)
                     + 'target_acc_' + str(self.args.target), 'w').writelines(losses)
                print('Round {:3d}, Average loss {:.4f}'.format(i, test_loss))
                break
    

    # test the trained model with test data
    def test(self):
        self.server_model.eval()
        test_data_loader = DataLoader(self.test_data, batch_size=self.args.test_batch_size)
        test_loss = 0
        correct = 0
        for idx, (data, target) in enumerate(test_data_loader):
            if self.args.gpu != -1 and torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.server_model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).sum()
        test_loss /= len(test_data_loader.dataset)
        accuracy = 100.00 * correct.float() / len(test_data_loader.dataset)
        if self.args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, len(test_data_loader.dataset), accuracy))
        return accuracy, test_loss
