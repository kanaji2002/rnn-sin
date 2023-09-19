import torch.nn as nn
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

sin_x=torch.linspace(-2*math.pi, 2*math.pi, 100)
sin_y=torch.sin(sin_x)+0.1*torch.randn(len(sin_x))
plt.plot(sin_x, sin_y)
plt.show()

n_time=10
n_sample=len(sin_x)-n_time

input_data=torch.zeros((n_sample, n_time, 1))
correct_data=torch.zeros((n_sample,1))
for i in range(n_sample):
    input_data[i]=sin_y[i:i+n_time].view(-1,1)
    correct_data[i]=sin_y[i+n_time:i+n_time+1]
    
dataset=TensorDataset(input_data, correct_data)
train_loader=DataLoader(dataset,batch_size=8,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.RNN(
            input_size=1,
            hidden_size=64,
            batch_first=True,
        )
        self.fc=nn.Linear(64,1)
        
    def forward(self, x):
        y_rnn, h=self.rnn(x,None)
        y=self.fc(y_rnn[:,-1,:])
        return y

net=Net()
print(net)