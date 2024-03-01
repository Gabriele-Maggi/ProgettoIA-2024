import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium import spaces
import random
from numpy.random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
import math

class AddictiveEnv_v3(gym.Env):
    def __init__(self):
        # Initialize environment parameters
        
        # Number of states
        self.numero_stati = 5  # 0:x 1:H 2:N 3:A 4:C
        
        # Number of actions
        self.numero_azioni = 5  # as2 as3 aG aW aD
        
        # Initial state
        self.S0 = 2
        
        # Rewards for different states
        self.reward_healty = 1
        self.reward_addicted = 10
        self.reward_penality = -20
        
        # Penalties
        self.C_penality = 0  # Reward for standing still in aftereffects    
        
        # Define observation and action space
        self.observation_space = spaces.Discrete(self.numero_stati)  # Not using state 0 
        self.action_space = spaces.Discrete(self.numero_azioni)         
        self.state = self.S0

        # Environment phase
        # 0-50 safe -> addictive deactivated
        # 50-1000 safe -> addictive activated
        self.env_phase = 0 
        self.DINIT = 0
        self.DDRUG = 1000

        # Bandit parameters
        self.epsilon = 0     
        self.c = 1
        self.non_addictive_reward = -1
        
        self.arms = 2
        
        # Initialize bandit parameters
        self.number_action = np.ones(self.arms)
        self.reward_action = np.ones(self.arms)
        self.q_bandit = np.zeros(self.arms)
        
        self.t = 1  # Updated with each action taken by the agent
        
        self.statistics = []    # [time step, chosen arm]
      
        self.current_arm = 0  # 0:4 1:2

    def set_reward(self, reward):
        # Set non-addictive reward
        self.non_addictive_reward = reward
        
    def get_rewards(self):
        return self.reward_action
    
    def get_arms(self):
        return self.current_arm
        
    def get_statistics(self):
        return self.statistics
            
    def get_iter(self):
        return self.DINIT + self.DDRUG
        
    def _get_obs(self):
        return self.state
    
    def _calculate_rew(self):
        # Calculate reward for each arm
        for ar in range(self.arms):
            self.q_bandit[ar] = self.reward_action[ar] / self.number_action[ar] 
    
    def _calculate_action(self):
        # Calculate action based on bandit algorithm
        self._calculate_rew()
        temp = np.zeros(self.arms)
        for ar in range(self.arms):
            temp[ar] = (self.q_bandit[ar] + (self.c * math.sqrt(math.log(self.t) / self.number_action[ar] )))
       
        new_arm = np.argmax(temp)
            
        if self.current_arm != new_arm:
            self.t += 1
        self.current_arm = new_arm
        
        if random.uniform(0, 1) < self.epsilon: # epsilon greedy 
            self.current_arm = random.randint(0, 1)
        
    def step(self, action):
        # Execute one time step within the environment
        
        # Adjust action space based on environment phase
        if (self.env_phase < self.DINIT):
            self.action_space = spaces.Discrete(self.numero_azioni - 1)
        else:
            self.action_space = spaces.Discrete(self.numero_azioni)
            
        reward = 0

        if self.state == 1:
            if action == 2:
                self.state = self.S0
                reward = self.reward_healty
        elif self.state == 2:    
            if action == 2:
                self.state = 1
            elif action == 1:
                self.state = 3
        
        elif self.state == 3:
            if action == 0:
                self.state = 2
                
                if self.env_phase > self.DINIT:
                    self.reward_action[self.current_arm] -= 1
                    self._calculate_action()
            
            elif action == 4:
                
                self.reward_action[self.current_arm] += 1
                self.number_action[self.current_arm] += 1
                self._calculate_action()
                if self.current_arm == 0:
                    self.state = 4
                    reward = self.reward_addicted       
                elif self.current_arm == 1:
                    self.state = 2
                    reward = self.non_addictive_reward 
            else:
                if self.env_phase > self.DINIT:
                    self.reward_action[self.current_arm] -= 1
                    self._calculate_action()
            
                
        elif self.state == 4:
            if action == 3:
                self.state = self.S0
                reward = self.reward_penality
            else:
                reward = self.C_penality   
                
        if (self.env_phase == self.DDRUG + self.DINIT):
            terminated = True
        else:
            terminated = False
            self.env_phase += 1  

        self.statistics.append(self.current_arm)
        return self._get_obs(), reward, terminated, False, {}
    
    def render(self):
        pass

    def reset_bandit(self):
        # Reset bandit parameters
        self.statistics = []
        self.t = 1 
        self.current_arm = 0  # 0:4 1:2
        self.number_action = np.ones(self.arms)
        self.reward_action = np.ones(self.arms)
        self.q_bandit = np.zeros(self.arms)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset the environment to initial state
        
        super().reset(seed=seed)
        self.reset_bandit()
        self.state = self.S0
        self.env_phase = 0 
        
        return self._get_obs(), {}
        
    def close(self):
        pass 