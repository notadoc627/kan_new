import torch
import torch.nn as nn

# 定义 MLP 模型
class MnistMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MnistMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CIFARMLP(nn.Module):
    def __init__(self,in_dim=3*32*32):
        super(CIFARMLP,self).__init__()
        self.lin1 = nn.Linear(in_dim,128)
        self.lin2 = nn.Linear(128,64)
        self.lin3 = nn.Linear(64,10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = x.view(-1,3*32*32)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        #x = self.relu(x)
        return x


                           