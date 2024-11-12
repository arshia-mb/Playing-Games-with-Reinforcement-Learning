import numpy as np
import torch


Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)

#Tree structure for storing the memory data
class SumTree:
    def __init__(self, capacity):
        self.size = capacity
        self.tree_start = 2**(capacity-1).bit_length() - 1
        self.tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.empty(capacity, dtype=Transition_dtype)
        self.index = 0
        self.max = 1.0  # Initial max value, set to a reasonable value
        self.full = False

    #update the value of the tree given the index
    def update(self, tree_index, value):
        change = value - self.tree[tree_index]
        self.tree[tree_index] = value
        self._propagate(tree_index, change)
        self.max = max(value, self.max)

    #propagate changes up the tree
    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    #Add new value and data to the tree
    def add(self, value, data):
        tree_index = self.index + self.tree_start
        self.data[self.index] = data #add the new data
        self.update(tree_index, value) #update the tree
        self.max = max(self.max, value)
        #update tree information
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0

    #Total sum of values in the tree
    def get_sum(self):
        return self.tree[0]

    #retrive tree node based on the value
    def retrieve(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.tree_start
        return self.tree[leaf_index], data_index, leaf_index
    
    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    #total sum of the values in the tree
    def total(self):
        return self.tree[0]

#Prioruty Experince Replay - Proportional Prioritization 
class ReplayMemory():
    def __init__(self, args, capacity):
        self.capacity = capacity
        self.device = args.device
        self.discount = args.discount
        self.n = args.multi_step
        self.history = args.frame_stack
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.beta_increment = args.beta_increment
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device) # Discount-scaling vector for n-step returns
        self.transitions = SumTree(capacity)
        self.t = 0 #Internal episode timestep counter

    #add new transition
    def append(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        self.transitions.add(self.transitions.max,(self.t, state,action,reward, not terminal)) #store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1 #Start new episode with t = 0

    def _get_transitions(self, idxs):
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1) #indexes for all the needed transitions (n-step return calcualtions)
        transitions = self.transitions.get(transition_idxs) #get data from the tree
        transitions_firsts = transitions['timestep'] == 0 #create a list where the episode has started
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1]) # True if future frame has timestep 0 - end of episode
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0 - start of new episode
        transitions[blank_mask] = blank_trans #make the transitions that are in the end of episodes blank
        return transitions


    #sample batch of transitions from memory based on their priority
    def sample(self, batch_size):
        p_total = self.transitions.total() #Total probebility 

        #getting sample transitions from the tree
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        temp = 0
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            prios = np.empty(batch_size, dtype=np.float32)  #prioriy sampled from each segment
            idxes = np.empty(batch_size, dtype=np.int64) #index of the data - for getting transition
            tree_idxes = np.empty(batch_size, dtype=np.int64) #tree_index for updaing priority values
            i = 0
            #selecting uniformly from each segment
            for sample in samples:
                priority, data_index, index  = self.transitions.retrieve(sample)
                prios[i] = (priority)  #priority
                idxes[i] = (data_index)#data index in the data array
                tree_idxes[i] = (index)#tree index
                i += 1
            #check if the selected transitions are okay
            temp += 1
            if np.all((self.transitions.index - idxes) % self.capacity > self.n) and np.all((idxes - self.transitions.index) % self.capacity >= self.history) and np.all(prios != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
            if temp >= 50:
                valid = True
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxes)
        # Create un-discretised states and nth next states
        all_states = transitions['state']
        states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
        next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255)
        
        # Discrete actions to be used as index
        actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
        
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        
        # Mask for non-terminal nth next states
        nonterminals = torch.tensor(np.copy(transitions['nonterminal'][:, self.history + self.n - 1]), dtype=torch.float32, device=self.device)
        
        probs = prios / p_total #normalize probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index #necessary for when memory is not full
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch 
        
        return tree_idxes, states, actions, R, next_states, nonterminals, weights

    #update priorities based on TD errors
    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent) #the exponent priority is saved
        for index, value in zip(idxs,priorities):
            self.transitions.update(index,value)

    #Aneeling importance-sampling weight to 1    
    def increment_beta(self):
        self.priority_weight = min(1.0, self.priority_weight + self.beta_increment)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # If future frame has timestep 0
        transitions[blank_mask] = blank_trans
        state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state