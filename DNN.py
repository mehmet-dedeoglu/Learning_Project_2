import torch.nn as nn
import torch.nn.functional as F


class DNN_v1(nn.Module):
    def __init__(self):
        super(DNN_v1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 10)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x


class DNN_v2(nn.Module):
    def __init__(self):
        super(DNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DNN_v3(nn.Module):
    def __init__(self):
        super(DNN_v3, self).__init__()
        self.fc2 = nn.Linear(32 * 32, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# # plt.switch_backend('agg')
#
#
# class DNN_v1(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.main_module = nn.Sequential(
#             nn.Linear(channels, 256),
#             nn.ReLU(),
#
#             nn.Linear(channels, 64),
#             nn.ReLU(),
#
#             nn.Linear(channels, 10),
#             nn.ReLU())
#
#         self.output = nn.Tanh()
#
#     def forward(self, x):
#         new_shape = x.shape[1]*x.shape[2]*x.shape[3]
#         y = torch.reshape(x, new_shape)
#         y = self.main_module(y, new_shape)
#         return self.output(y)
