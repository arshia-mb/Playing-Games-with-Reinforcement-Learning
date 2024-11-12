import math
import torch
from torch import nn
from torch.nn import functional as F

#Factorised NoisyLinear layer with bias - based on Rainbow implementation 
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init /(self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
    

#Dueling DQN network using NoisyLinear layers
class DDQN(nn.Module):
    def __init__(self, args, action_space):
        super(DDQN,self).__init__()
        self.action_Space = action_space

        #feature extraction layers
        self.convs = nn.Sequential(nn.Conv2d(args.frame_stack, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU()) 
        self.conv_output_size = 3136
        #value function stream
        self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
        #advantage function stram
        self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_a = NoisyLinear(args.hidden_size, self.action_Space, std_init=args.noisy_std)
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        v = self.fc_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        q = v + a - a.mean(1, keepdim=True) # Combine streams
        return q
    
    def reset_noise(self): #reset noise on the NoisyLinear layers
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()