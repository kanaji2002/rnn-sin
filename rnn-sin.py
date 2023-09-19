import torch.nn as nn
import math
import torch
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ...
        self.RNN(
            input_size=1,
            hidden_size=64,
            batch_first=True,
        )
        ...
        
    def forward(self, x):
        ...

        y_rnn, h=self.rnn(x,None)
        ...
        return y

sin_x=torch.linspace(-2*math.pi, 2*math.pi, 100)
sin_y=torch.sin(sin_x)
plt.plot(sin_x, sin_y)
plt.show()

       
               