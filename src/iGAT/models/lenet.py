import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.Linear = nn.Linear(4 * 4 * 50, 500)
        self.input_vec = nn.Linear(500, 64)
        self.predicted_label = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.Linear(x)

        x = F.relu(x)
        x = self.input_vec(x)

        x = F.relu(x)
        x = self.predicted_label(x)

        return x

    def get_features(self, x, layer, before_relu=False):
        x = F.relu(self.conv1(x))
        if layer == 1:
            return x

        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if layer == 2:
            return x

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.Linear(x)
        x = F.relu(x)
        if layer == 3:
            return x

        x = self.input_vec(x)
        x = F.relu(x)
        if layer == 4:
            return x

        x = self.predicted_label(x)
        return x

    def infer(self, x):
        return self.forward(x).max(1, keepdim=True)[1]

    def get_input_vec(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.Linear(x)
        x = F.relu(x)
        x = self.input_vec(x)
        return x

    def get_predicted_label(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.Linear(x)
        x = F.relu(x)
        x = self.input_vec(x)
        x = F.relu(x)
        x = self.predicted_label(x)
        return x
