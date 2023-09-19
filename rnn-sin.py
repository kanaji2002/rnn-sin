import torch.nn as nn
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
  

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

n_time=10
n_sample=len(sin_x)-n_time

input_data=torch.zeros((n_sample, n_time, 1))
correct_data=torch.zeros((n_sample,1))
for i in range(n_sample):
    input_data[i]=sin_y[i:i+n_time].view(-1,1)
    correct_data[i]=sin_y[i+n_time:i+n_time+1]
    
dataset=TensorDataset(input_data, correct_data)
train_loader=DataLoader(dataset,batch_size=8,shuffle=True)

class net(nn.Module):
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

loss_fnc=nn.MSELoss()

optimizer=optim.SGD(net.parameters(), lr=0.01)
#学習率は0.01

#損失のログ
record_loss_train=[]

#学習
epochs=100
for i in range(epochs):
    net.train() #訓練モード
    loss_train=0
    for j, (x,t) in enumerate(train_loader):
        #ミニバッチ(x,t)を取り出す
        y=net(x)
        loss=loss_fnc(y,t)
        loss_train+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    loss_train/=j+1
    record_loss_train.append(loss_train)
    
    #経過の表示
    if i%10==0 or i==epochs-1:
        net.eval()
        print("Epoch:",i,"Loss_Train:",loss_train)
        predicted=list(input_data[0].view(-1))
        
#最後の入力
        for i in range(n_sample):
            x=torch.tensor(predicted[-n_time:])
#直近の時系列を取り出す
            y=net(x)
            predicted.append(y[0].item())
#予測結果をpredictedに追加する
        plt.plot(range(len(sin_y)),sin_y,label="Correct")
        plt.plot(range(len(predicted)),predicted,label="Predicted")
        plt.legend()
        plt.show()



