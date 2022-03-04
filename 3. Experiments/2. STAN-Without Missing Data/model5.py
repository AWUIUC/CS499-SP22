import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
#warnings.filterwarnings('ignore')

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))
        
class STAN_v5(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device):
        super(STAN_v5, self).__init__()
        self.g = g
        
        self.layer1 = MultiHeadGATLayer(self.g, in_dim, hidden_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(self.g, hidden_dim1 * num_heads, hidden_dim2, 1)

        self.pred_window = pred_window
        self.gru = nn.GRUCell(hidden_dim2, gru_dim)
    
        self.nn_cumulative_confirmed = nn.Linear(gru_dim, pred_window)
        
        self.hidden_dim2 = hidden_dim2
        self.gru_dim = gru_dim
        self.device = device
        

    def forward(self, dynamic, h=None):
        num_loc, timestep, n_feat = dynamic.size()

        if h is None:
            h = torch.zeros(1, self.gru_dim).to(self.device)
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(h, gain=gain)  

        cumulative_confirmed = []

        debug_messages = []

        for each_step in range(timestep):        
            cur_h = self.layer1(dynamic[:, each_step, :])
            cur_h = F.elu(cur_h)
            cur_h = self.layer2(cur_h)
            cur_h = F.elu(cur_h)
            
            debug_messages = []
            debug_messages.append('hello_world')
            debug_messages.append(cur_h)
            
            cur_h = torch.max(cur_h, 0)[0].reshape(1, self.hidden_dim2)
            
            self.debug_messages.append(cur_h)
            
            h = self.gru(cur_h, h)
            
            predicted_confirmed = self.nn_cumulative_confirmed(h)
            
            cumulative_confirmed.append(predicted_confirmed)
            
            debug_messages.append(predicted_confirmed)
            

        cumulative_confirmed = torch.stack(cumulative_confirmed).to(self.device).permute(1,0,2)

        debug_messages = debug_messages.to(self.device)

        # return cumulative_confirmed, cumulative_deaths, h
        return cumulative_confirmed, h, debug_messages