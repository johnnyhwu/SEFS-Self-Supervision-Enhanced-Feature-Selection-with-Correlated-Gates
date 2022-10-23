import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.)
    
    def forward(self, input):

        x = self.linear1(input)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.linear2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.linear3(x)
        x = F.relu(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(10, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.dropout = nn.Dropout(0.)
    
    def forward(self, input):
        x = self.linear1(input)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.linear2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x

class Phase_1_Model(torch.nn.Module): 
    def __init__(self):   
        super(Phase_1_Model, self).__init__()

        # build encoder and two decoder
        self.enc = Encoder()
        self.dec_x = Decoder()
        self.dec_m = Decoder()
        
    def forward(self, x, x_bar, mask):
        # feature-selected x
        x_tilde = (x * mask) + (x_bar * (1-mask))

        z = self.enc(x_tilde)
        x_hat = self.dec_x(z)
        m_hat = self.dec_m(z)
        return x_hat, m_hat