import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, state_size, num_passengers, mix_hidden):
        super(QMixer, self).__init__()

        self.state_size = state_size
        self.num_passengers = num_passengers
        self.mix_hidden = mix_hidden
        
        self.hyper_w_1 = nn.Linear(state_size, num_passengers * mix_hidden)
        self.hyper_w_final = nn.Linear(state_size, mix_hidden)
        # elif getattr(args, "hypernet_layers", 1) == 2:
        #     hypernet_embed = self.args.hypernet_embed
        # self.hyper_w_1 = nn.Sequential(nn.Linear(state_size, 32),
        #                                 nn.ReLU(),
        #                                 nn.Linear(32, mix_hidden * num_passengers))
        # self.hyper_w_final = nn.Sequential(nn.Linear(state_size, 32),
        #                                 nn.ReLU(),
        #                                 nn.Linear(32, mix_hidden))
        
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(state_size, mix_hidden)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(state_size, mix_hidden),
                               nn.ReLU(),
                               nn.Linear(mix_hidden, 1))

    def forward(self, agent_qs, states):
        
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_size)
        agent_qs = agent_qs.view(-1, 1, self.num_passengers)

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.num_passengers, self.mix_hidden)
        b1 = b1.view(-1, 1, self.mix_hidden)
        
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.mix_hidden, 1)
        
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        
        return q_tot