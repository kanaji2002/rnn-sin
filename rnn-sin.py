import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ...
        self.RNN(
            input_size=1,
            hidden_size=64,
            batch_first=True,
        )