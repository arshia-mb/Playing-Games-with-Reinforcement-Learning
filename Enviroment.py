import gymnasium as gym
from gymnasium import spaces

class Enviroment:
    def __init__(self, args, eval):
        self.env = gym.make(args.game, render_mode = args.render_mode)
        self._preprocess(args.noop, args.frame_stack, args.frame_skip, eval, args.max_episode_length)

    #adding noop, framestacking, grayscaling and resizing the input pixels
    def _preprocess(self, noop, frame_stack, frame_skip, eval, max_episode_steps):
        env = gym.wrappers.TimeLimit(self.env, max_episode_steps) #truncated after a number of episodes
        atari = gym.wrappers.AtariPreprocessing(env, noop_max = noop, frame_skip = frame_skip, screen_size = 84, 
                        terminal_on_life_loss=(not eval), grayscale_obs = True) #gray sacling and donw samplling 
        frame_stacked = gym.wrappers.FrameStack(atari, frame_stack) #stacking last 4 frames
        #env = FireResetWrapper(frame_stacked)
        self.env = frame_stacked

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space.n
    
    @property
    def shape(self):
        return self.env.observation_space.shape
    
class FireResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FireResetWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(self.action_space.n + 1)  # Add one more action for FIRE

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation, _, _, _ = self.env.step(self.action_space.n)  # Perform FIRE action
        return observation
