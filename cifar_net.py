import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

OMEGA = 1
HIDDEN1 = 120
HIDDEN2 = 84
CONV_OUT = 16 * 5 * 5
NUM_CLASSES = 10

class Net(nn.Module):
    def __init__(self, net_type):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(CONV_OUT, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, NUM_CLASSES)

        self.fc1n = nn.Linear(CONV_OUT, HIDDEN1)
        self.fc2n = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3n = nn.Linear(HIDDEN2, NUM_CLASSES)

        self.net_type = net_type

    def forward(self, x):
        # conv block:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, CONV_OUT)

        # fc block:
        if 'synergy' in self.net_type:
            x_normal = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
            x_negative = self.fc3n(F.relu(self.fc2n(
                F.relu(self.fc1n(1 - x)))))

            x = x_normal + x_negative * OMEGA

        else:
            if 'hybrid' in self.net_type:
                x = 1 - x

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        return x
