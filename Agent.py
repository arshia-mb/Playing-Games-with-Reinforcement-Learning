import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from Model import DDQN

class Agent():
    def __init__(self, args, action_space):
        self.action_space = action_space
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.epsilon = args.epsilon

        self.online_net = DDQN(args, self.action_space).to(device=args.device)
        self.online_net.train() #online network
        self.target_net = DDQN(args, self.action_space).to(device=args.device)
        self.update_target_net() #target network for evaluations 
        self.target_net.eval()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate) #optimizer

    #select action based on the input state
    def select_action(self, state, greedy=False):
        if greedy and np.random.random() < self.epsilon: #Acts with an ε-greedy policy
            return np.random.randint(0, self.action_space) 
        else: #when training don't need to use ε, NoisyLinear layers solve exploration problem
            with torch.no_grad():
                return self.online_net(state.unsqueeze(0)).argmax(1).item()
            
    def learn(self, mem, batch_size):
        #sample transitions
        idxes, states, actions, returns, states_, nonterminals, weights = mem.sample(batch_size)

        with torch.no_grad():
            next_state_values = self.target_net(states_).max(1)[0]  #Calcualte the nth next state values using the target network
            target_values = returns + nonterminals * (self.discount ** self.n) * next_state_values  #Calculate the target values for the current states

        # Calculate Q-values for the current states using the online network
        current_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Calculate the mean squared TD error
        loss = F.mse_loss(current_values, target_values, reduction='none')

        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()
        
        mem.update_priorities(idxes, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    #Evaluates Q-value based on single state
    def evaluate_q(self, state):
        with torch.no_grad():
            return  self.online_net(state.unsqueeze(0)).max(1)[0].item()


    #resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    #update the target network to online network
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on device
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Load model from the the device 
    def load(self, path, name='model.pth'):
        state_dict = torch.load(os.path.join(path, name), map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        self.online_net.load_state_dict(state_dict)

    def train(self): #set network to train mode
        self.online_net.train()

    def eval(self): #set metwprl to evaluation mode
        self.online_net.eval()