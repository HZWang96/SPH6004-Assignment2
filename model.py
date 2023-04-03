from torch import nn
from torch.nn.modules.activation import Softmax
from torch.nn.modules.linear import Linear
from torch.nn import LSTM 
import torch
class lstm(nn.Module):
    def __init__(self,hidden_size,input_size,batch_size):
        super(lstm,self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.layer = LSTM(hidden_size=hidden_size, input_size=input_size,batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden_size,10),nn.Linear(10,1))
        self.loss = nn.MSELoss()
        for param in self.parameters():
            param.data = param.data.to(torch.float64)
    def forward(self, X):
        h_0 = torch.randn(1,self.batch_size,self.hidden_size).cuda().to(torch.float64)
        c_0 = torch.randn(1,self.batch_size,self.hidden_size).cuda().to(torch.float64)
        out, _ = self.layer(X,(h_0,c_0))
        out = self.output(out)
        return out
