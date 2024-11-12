import torch
import numpy as np
import os
from types import SimpleNamespace
from tqdm import trange
from Enviroment import Enviroment
from Agent import Agent

args = SimpleNamespace(
        id='enduro', #Experiment id
        seed=123, #Random seed for numpy
        game='EnduroNoFrameskip-v4', #Name of the game
        max_episode_length=int(108e3), #Max episode length in game frames
        frame_skip=4,  #Number of frames skipped (k) 
        frame_stack=4, #Number of consecutive states processed (T)
        noop=30, #Max number of no-op actions taken at the start of each episode
        hidden_size = 512, #Number of the neurons in the hidden layers
        noisy_std = 0.1, #Initial standard deviation of nosy linear layers (σ)
        model=None, #Pretrained model - path to the model
        memory_capacity=int(1e6), #Experience replay memory capacity 
        max_epoch=int(50e6), #Number of training steps
        replay_frequency=4, #Frequency of sampling from memory (k)
        priority_exponent=0.6, #experience replay exponent (α)
        priority_weight=0.4,   #experience replay initial importance sampling (β)
        multi_step = 3, #Number of steps for multi-step returns (n)
        discount=0.99, #Q-learning discount factor (γ)
        target_update=int(8e3),#Number of steps after which to update target network (τ)
        reward_clip=1,  #Reward Clipping
        norm_clip=10, #Norm for gradient clipping
        learning_rate=0.0000625, #Learning rate (η)
        batch_size=32,#Batch size for training 
        learn_start=int(20e3), #Number of steps before starting training 
        eval_interval=int(1e6),#Number of training steps between evaluations   
        eval_episodes=4, #Number of evaluation episodes to average over
        eval_size=100, #Number of transitions to use for validating Q
        epsilon=0.001, #Used for ε-greedy actions (not needed when using NoisyNet)    
        render_mode='human' #for test only
    )

results_dir = os.path.join('results', 'final_models') 
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000)) #random seed for cpu
if torch.cuda.is_available() :
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000)) #random seed for gpu
else:
  args.device = torch.device('cpu')

args.beta_increment = (1 - args.priority_weight) / (args.max_epoch - args.learn_start) #Aneeling steps for β to 1

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')} #test results for each test

# Environment setup
env = Enviroment(args, eval=False) #enviroment in training mode
agent = Agent(args,env.action_space)
agent.load(results_dir, args.id + "_model.pth")
agent.eval() #set agent to evaliation mode

test_rewards = []
#Test the performance over several episodes
done = True
for _ in trange(args.eval_episodes):
  while True:
    if done or truncated: #resetting the enviroment
      state, _ = env.reset()
      r_sum = 0
      done = False
            
    state = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float32, device=args.device).div_(255) #the input for the network is a torch tensor
    action = agent.select_action(state, greedy=True) #select greedy action
    state, reward, done, truncated, _ = env.step(action)
    r_sum += reward
            
    if done or truncated: #checks if the number of frames in the episode has exceeded the limit or the episode is done
      test_rewards.append(r_sum)
      break

env.close()
print("test is concluded")
print("\n Average reward:" + str(sum(test_rewards) / len(test_rewards)))
print("\n")




