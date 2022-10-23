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

class Predictor(torch.nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 2)
    
    def forward(self, input):
        x = self.linear1(input)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=-1)
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

class Phase_2_Model(torch.nn.Module): 
    def __init__(self, enc_path):   
        super(Phase_2_Model, self).__init__()

        self.pi_logit = nn.Parameter(torch.zeros((100, )))
        self.predictor = Predictor()

        self.enc = Encoder()
        self.enc.load_state_dict(torch.load(enc_path, map_location="cpu"))
        for param in self.enc.named_parameters():
            if "linear3" not in param[0]:
                param[1].requires_grad = False

    def gaussian_cdf(self, x):
        return 0.5 * (1. + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
    def log(self, x): 
        return torch.log(x + 1e-8)
        
    def forward(self, x, q):
        self.pi = torch.sigmoid(self.pi_logit)
        u = self.gaussian_cdf(q)
        m = torch.sigmoid(self.log(self.pi) - self.log(1.0 - self.pi) + self.log(u) - self.log(1.0 - u))
        x_tilde = x * m
        z = self.enc(x_tilde)
        out = self.predictor(z)
        return out
