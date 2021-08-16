from torch import nn
import torch.nn.functional as F
import torch

# Define the residual block
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(4096,4096)
        self.bn1 = nn.BatchNorm1d(num_features=4096)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        residual = x
        out = self.linear(x)
        out = F.relu(self.bn1(out))
        out = self.dropout(out)
        # out = self.linear(x)
        out += residual
        return out

# Define the main network
class BodyNetwork(nn.Module):
    def __init__(self):
        super(BodyNetwork, self).__init__()
        self.fc=torch.nn.Linear(54*2,4096)
        self.resblock=ResBlock()
        self.fc1=torch.nn.Linear(4096,14*4)
    def forward(self,x):
        x=x.view(-1,54*2)
        out=self.fc(x)
        out=self.resblock(out)
        # out=self.resblock(out)
        out=self.fc1(out)
        return torch.tanh(out)

class HandNetwork(nn.Module):
    def __init__(self):
        super(HandNetwork, self).__init__()
        self.fc=torch.nn.Linear(49*2,4096)
        self.resblock=ResBlock()
        self.fc1=torch.nn.Linear(4096,38*4)
        self.dropout=torch.nn.Dropout(0.5)
    def forward(self,x):
        x=x.view(-1,49*2)
        out=self.fc(x)
        out=self.resblock(out)
        out=self.fc1(out)
        return torch.tanh(out)