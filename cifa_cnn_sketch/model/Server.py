import copy
import _pickle as pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .Sketch import Sketch

from model.Network import NN, MLP_SketchLinear, CNNCifar, CNNMnist, CNNMnist_Sketch, CNNCifar_Sketch


class Server:
    # build a server
    # broadcast parameters and sketch matrix S to clients
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
        print('Whether run on GPU', next(self.server_model.parameters()).is_cuda)
        self.global_weights = copy.deepcopy(self.server_model.state_dict())
        if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
            self.sizes = self.server_model.weight_sizes()

    # collect local gradients from the selected clients in each communicaiton rounds
    def get_grads(self):
        self.local_grads = [self.clients[client_id].send_grads() for client_id in self.working_client]

    # randomly select some clients for training in each communication rounds
    # broadcast global gradients and sketch matrix to all selected clients
    def broadcast(self):
        # randomly generate sketch matrix in a communication round
        if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
            hash_idxs = []
            rand_sgns = []
            for size in self.sizes:
                hash_idx, rand_sgn = Sketch.rand_hashing(size, q=self.args.p)
                hash_idxs.append(hash_idx)
                rand_sgns.append(rand_sgn)
            num_client = int(len(self.clients) * self.args.sample_rate)
            # select working client in a round
            self.working_client = np.random.choice(len(self.clients), num_client, replace=False)
            for client_id in self.working_client:
                self.clients[client_id].get_paras(copy.deepcopy(self.global_weights), hash_idxs, rand_sgns)
            return hash_idxs, rand_sgns
        else:
            num_client = int(len(self.clients) * self.args.sample_rate)
            self.working_client = np.random.choice(len(self.clients), num_client, replace=False)
            # self.working_client = [0]
            for client_id in self.working_client:
                self.clients[client_id].get_paras(copy.deepcopy(self.global_weights), None, None)

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

    def update_paras(self):
        self.get_grads()
        g_avg = self.average_grads()
        for k in g_avg.keys():
            self.global_weights[k] = self.global_weights[k] + g_avg[k]
        self.server_model.load_state_dict(self.global_weights)

    # server updates the model parameters using averaged gradients
    def average_grads(self):
        return self._average(self.local_grads)


    def w_err_server(self, w_old, w_new, hash_idxs_old, rand_sgns_old):
        err = 0
        i = 0
        delta_ws, delta_tilde_ws = [], []
        start_index = 1 if self.args.model_type == 'CNN_sketch' and self.args.datatype == 'cifar' else 0
        w_old_list = list(w_old.values())[start_index:]
        w_new_list = list(w_new.values())[start_index:]
        for w_o, w_n in zip(w_old_list, w_new_list):
            if len(w_o.shape) == 1:
                continue
            if i < len(hash_idxs_old):
                w_o_sketch = Sketch.countsketch(w_o.to(self.args.device), hash_idxs_old[i], rand_sgns_old[i]).to(
                    self.args.device)
                w_n_sketch = Sketch.countsketch(w_n.to(self.args.device), hash_idxs_old[i], rand_sgns_old[i]).to(
                    self.args.device)
                delta_w = w_o - w_n
                # temp = delta_w.detach().numpy()
                # pickle.dump([temp, hash_idxs_old[i], rand_sgns_old[i]], open('temp', 'wb'));exit()
                delta_w_client = w_o_sketch - w_n_sketch

                delta_tilde_w = 0.5*Sketch.transpose_countsketch(delta_w_client.to(self.args.device), hash_idxs_old[i],
                                                             rand_sgns_old[i]).to(self.args.device)

                delta_ws.append(torch.reshape(delta_w, (-1,)))
                delta_tilde_ws.append(torch.reshape(delta_tilde_w, (-1,)))
                # print('a', torch.norm(delta_tilde_w - delta_w) / torch.norm(delta_w))
                i += 1
        delta_ws_v = torch.cat(delta_ws, dim=-1)
        delta_tilde_ws_v = torch.cat(delta_tilde_ws, dim=-1)
        a = torch.norm(delta_ws_v - delta_tilde_ws_v)
        b = torch.norm(delta_ws_v)
        err = a / b
        return err

    # compute client side gradient approximation error ratio with the transpose of S, S^T or pseudo-inverse of S, S^+
    # S^+ = 1/p * S^T, p is the the parameters of Sketch
    # the metrics we used are F-norm and cosine similarity

    def w_err_client(self, w_old, w_new, hash_idxs_old, rand_sgns_old, hash_idxs_new, rand_sgns_new):
        err = 0
        i = 0
        delta_ws, delta_tilde_ws, delta_tilde_ws_scale = [], [], []
        start_index = 2 if self.args.model_type == 'CNN_sketch' and self.args.datatype == 'cifar' else 0
        w_old_list = list(w_old.values())[start_index:]
        w_new_list = list(w_new.values())[start_index:]
        for w_o, w_n in zip(w_old_list, w_new_list):
            if len(w_o.shape) == 1:
                continue
            if i < len(hash_idxs_old):
                w_o_sketch = Sketch.countsketch(w_o.to(self.args.device), hash_idxs_old[i], rand_sgns_old[i]).to(
                    self.args.device)
                w_o_tran_sketch = Sketch.transpose_countsketch(w_o_sketch.to(self.args.device), hash_idxs_old[i],
                                                               rand_sgns_old[i]).to(self.args.device)
                w_n_sketch = Sketch.countsketch(w_n.to(self.args.device), hash_idxs_new[i], rand_sgns_new[i]).to(
                    self.args.device)
                w_n_tran_sketch = Sketch.transpose_countsketch(w_n_sketch.to(self.args.device), hash_idxs_new[i],
                                                               rand_sgns_new[i]).to(self.args.device)
                delta_w = w_o - w_n
                delta_tilde_w = w_o_tran_sketch - w_n_tran_sketch
                delta_tilde_w_scale = 0.5 * delta_tilde_w
                delta_ws.append(torch.reshape(delta_w, (-1,)))
                delta_tilde_ws.append(torch.reshape(delta_tilde_w, (-1,)))
                delta_tilde_ws_scale.append(torch.reshape(delta_tilde_w_scale, (-1,)))
                i += 1
        delta_ws_v = torch.cat(delta_ws, dim=-1)
        delta_tilde_ws_v = torch.cat(delta_tilde_ws, dim=-1)
        delta_tilde_ws_scale_v = torch.cat(delta_tilde_ws_scale, dim=-1)
        err = torch.norm(delta_ws_v - delta_tilde_ws_v) / torch.norm(delta_ws_v)
        err_scale = torch.norm(delta_ws_v - delta_tilde_ws_scale_v) / torch.norm(delta_ws_v)
        sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        err_cosine = sim(delta_tilde_ws_scale_v, delta_ws_v)
        return err, err_scale, err_cosine

    # clients train the local models on local data then send the gradients to server
    # server aggregates local gradients and updates the global model then sends global parameters to clients
    # server tests the global model after each communication rounds
    def train(self):
        accs, losses, errs_client, errs_client_scale, errs_client_cosine = [], [], [], [], []
        round = self.args.round
        w_old, hash_idxs_old, rand_sgns_old = None, None, None
        for i in range(round):
            print('server round', i)
            if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
                w_new = copy.deepcopy(self.global_weights)
                hash_idxs_new, rand_sgns_new = self.broadcast()
                if i >= 1:
                    w_error, w_err_scale, w_err_cosine = self.w_err_client(w_old, w_new, hash_idxs_old, rand_sgns_old, hash_idxs_new, rand_sgns_new)
                    # w_error_server = self.w_err_server(w_old, w_new, hash_idxs_old, rand_sgns_old)
                    errs_client.append(w_error.detach().cpu().numpy())
                    errs_client_scale.append(w_err_scale.detach().cpu().numpy())
                    errs_client_cosine.append(w_err_cosine.detach().cpu().numpy())
                    # errs_server.append(w_error_server.detach().cpu().numpy())
                    print('client side weight error:', w_error.detach().cpu().numpy())
                    print('client side weight error scale:', w_err_scale.detach().cpu().numpy())
                    print('client side weight error cosine:', w_err_cosine.detach().cpu().numpy())
                    # print('server sizde weight error:', w_error_server.detach().cpu().numpy())
                w_old = w_new
                hash_idxs_old, rand_sgns_old = hash_idxs_new, rand_sgns_new

                for client_id in self.working_client:
                    print('client', client_id)
                    client = self.clients[client_id]
                    train_loss, train_acc = client.train(i)
                    print('client', client_id, ' -- ', 'train loss:', train_loss, 'train_acc:', train_acc)
                self.update_paras()
                acc_test, test_loss = self.test()
                accs.append(acc_test)
                losses.append(test_loss)
                if acc_test >= self.args.target or i == (round-1):
                    pickle.dump(accs, open('data/results/accs_' + self.args.model_type + self.args.datatype + '_lr_' + str(
                        self.args.sample_rate) + 'target_acc_' + str(self.args.target), 'wb'))
                    pickle.dump(losses, open('data/results/losses_' + self.args.model_type + self.args.datatype + '_lr_' + str(
                        self.args.sample_rate) + 'target_acc_' + str(self.args.target), 'wb'))
                    pickle.dump(errs_client, open('data/results/w_errs_client' + self.args.model_type + self.args.datatype + '_lr_' + str(
                        self.args.sample_rate) + 'target_acc_' + str(self.args.target), 'wb'))
                    pickle.dump(errs_client_scale, open(
                        'data/results/w_errs_client_scale' + self.args.model_type + self.args.datatype + '_lr_' + str(
                            self.args.sample_rate) + 'target_acc_' + str(self.args.target), 'wb'))
                    pickle.dump(errs_client_cosine, open(
                        'data/results/w_errs_client_cosine' + self.args.model_type + self.args.datatype + '_lr_' + str(
                            self.args.sample_rate) + 'target_acc_' + str(self.args.target), 'wb'))
                    # pickle.dump(errs_server, open('data/results/w_errs_server' + self.args.model_type + self.args.datatype + '_lr_' + str(
                    #     self.args.sample_rate) + 'target_acc_' + str(self.args.target), 'wb'))
                    # print('Round {:3d}, Average loss {:.4f}, Weight error {:.4f}'.format(i, test_loss, w_error))
                    break
            else:
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
                if acc_test >= self.args.target or i==round:
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
            if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
                log_probs = self.server_model(data, [None] * len(self.sizes), [None] * len(self.sizes))
            else:
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
