import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
from torchvision import utils
import torch.optim as optim
from DNN import DNN_v3
import numpy as np


class learning_model(object):
    def __init__(self, args):
        print("Creating a new learning model...")
        self.DNN_model = DNN_v3()
        for p in self.DNN_model.parameters():
            p.requires_grad = True
        self.optim = optim.Adam(self.DNN_model.parameters(), lr=args.learning_rate,
                                betas=(args.beta_1, args.beta_2))
        self.cuda_index = 0
        self.cuda = False
        self.check_cuda(args.cuda)
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.loss_fn = nn.CrossEntropyLoss()
        self.grad_vector_final_shape = []
        self.grad_vector_initial_shape = []

    def get_torch_variable(self, inp):
        if self.cuda:
            return Variable(inp).cuda(self.cuda_index)
        else:
            return Variable(inp)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.DNN_model.cuda(self.cuda_index)
            print("Models are assigned to GPU : {}".format(self.cuda_index))
        else:
            self.cuda = False

    def grad_compute(self, samples, labels):

        self.DNN_model.train()
        self.DNN_model.zero_grad()
        torch_samples = self.get_torch_variable(samples)
        torch_labels = self.get_torch_variable(labels)
        torch_output = self.DNN_model(torch_samples)
        torch_cost = self.loss_fn(torch_output, torch_labels)
        torch_cost.backward()
        # grad_values = self.DNN_model.parameters()

        # Convert gradient matrices to a single column vector
        params = list(self.DNN_model.parameters())
        grad_vector = []
        self.grad_vector_final_shape = []
        self.grad_vector_initial_shape = []
        for p_ind in range(len(params)):
            temp_vector = params[p_ind].grad.cpu().numpy()
            initial_shape = temp_vector.shape
            new_shape = 1
            for j in range(len(initial_shape)):
                new_shape = new_shape*initial_shape[j]
            final_vector = temp_vector.reshape(new_shape)
            grad_vector.extend(final_vector)
            self.grad_vector_final_shape.append(new_shape)
            self.grad_vector_initial_shape.append(initial_shape)

        out_vector = np.asarray(grad_vector)

        return out_vector, self.grad_vector_final_shape, self.grad_vector_initial_shape

    def update_params(self, grad_param):
        self.DNN_model.train()
        grad_param_torch = self.get_torch_variable(torch.Tensor(grad_param))
        initial_index = 0
        params = list(self.DNN_model.parameters())
        with torch.no_grad():
            for ell in range(len(self.grad_vector_final_shape)):
                # p.grad = grad[i].clone()
                grads = grad_param_torch[initial_index:initial_index + self.grad_vector_final_shape[ell]]\
                    .reshape(self.grad_vector_initial_shape[ell]).clone()
                # params[ell].grad = grad_param_torch[initial_index:initial_index + self.grad_vector_final_shape[ell]]\
                #     .reshape(self.grad_vector_initial_shape[ell]).clone()
                initial_index = initial_index + self.grad_vector_final_shape[ell]
                new_val = params[ell] - grads
                params[ell].copy_(new_val)

        # print('params updated')
        # with torch.no_grad():
        #     for p in model.parameters():
        #         new_val = update_function(p, p.grad, loss, other_params)
        #         p.copy_(new_val)
        # # for p in self.DNN_model.parameters():
        # #     p.grad *=
        # self.optim.step()

    def check_accuracy(self, batch_test, labels_test, iteration):
        self.DNN_model.eval()
        correct = 0
        total = 0
        torch_samples = self.get_torch_variable(batch_test)
        torch_labels = self.get_torch_variable(labels_test)
        torch_output = self.DNN_model(torch_samples)

        _, predicted = torch.max(torch_output.data, 1)
        total += torch_labels.size(0)
        correct += (predicted == torch_labels).sum().item()
        accuracy = correct/total
        print('Accuracy at iteration ', str(iteration), ' is: ', str(accuracy))
        return accuracy













