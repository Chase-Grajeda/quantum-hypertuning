import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F

class LSTNet(nn.Module):
    def __init__(
        self,
        period: int, # ts length
        num_features: int, # feature dim
        rnn_dim: int = 50,
        cnn_dim: int = 50,
        skip_dim: int = 10,
        kernel: int = 4,
        num_skip: int = 2,
        attn_len: int = 4,
        dropout: float = 0.2
    ):
        super(LSTNet, self).__init__()
        self.period = period
        self.m = num_features
        self.hidR = rnn_dim
        self.hidC = cnn_dim
        self.hidS = skip_dim
        self.ck = kernel
        self.skip = num_skip
        self.attn = attn_len
        self.dropout = nn.Dropout(dropout)
                
        # CNN
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.ck, self.m))
        
        # RNN
        self.gru1 = nn.GRU(self.hidC, self.hidR)
        
        # SkipRNN
        self.pt = (self.period - self.ck) / self.skip
        self.gru_skip = nn.GRU(self.hidC, self.hidS)
        self.linear_skip = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        
        # Attention unit
        self.linear_attn = nn.Linear(self.attn, 1)
        
        # Output projection
        self.output = nn.Linear(self.m, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN pass
        c = x.view(-1, 1, self.period, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN pass
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.gru1(r)
        r = self.dropout(torch.squeeze(r, 0))
        
        # Skip RNN
        s = c[:, :, int(-self.pt * self.skip):].contiguous()
        s = s.view(batch_size, self.hidC, int(self.pt), self.skip)
        s = s.permute(2, 0, 3, 1).contiguous()
        s = s.view(int(self.pt), batch_size * self.skip, self.hidC)
        _, s = self.gru_skip(s)
        s = s.view(batch_size, self.skip * self.hidS)
        s = self.dropout(s)
        r = torch.cat((r, s), 1)
        
        # Flatten predictions
        u = self.linear_skip(r)
        
        # Apply attention
        z = x[:, -self.attn:, :]
        z = z.permute(0, 2, 1).contiguous().view(-1, self.attn)
        z = self.linear_attn(z)
        z = z.view(-1, self.m)
        
        # Perturb with residual attention
        res = self.output(u + z)
        
        return res
        