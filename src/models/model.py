from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3,
			kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5,
			kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=2000, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        x = F.log_softmax(self.fc2(x), dim=1)

        return x
