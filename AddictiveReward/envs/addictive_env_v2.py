import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium import spaces
import random
from numpy.random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional


class AddictiveEnv_v2(gym.Env):
    def __init__(self):
        
        self.numero_stati = 5 # 0:x 1:H 2:N 3:A 4:C
        self.numero_azioni = 5 # as2 as3 aG aW aD
        
        self.S0 = 2
        
        self.reward_healty = 1
        self.reward_addicted = 10
        self.reward_penality = -20
        
        self.C_penality = 0 # reward stando fermo in aftereffects    
        
        
        self.observation_space = spaces.Discrete(self.numero_stati) # non uso stato 0 
        self.action_space = spaces.Discrete(self.numero_azioni)         
        self.state = self.S0

        self.p = 1
                
        # env_phase: 0-50 safe -> addictive deactivated
        # env_phase: 50-1000 safe -> addictive activated
        self.env_phase = 0 
        self.DINIT = 0
        self.DDRUG = 200
        
        
    def get_iter(self):
        return self.DINIT + self.DDRUG

    def _get_obs(self):
        return self.state
    
    def step(self, action):
        
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
            elif action == 4:
                self.state = 4
                reward = self.reward_addicted                   
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
        
        return self._get_obs(), reward, terminated, False, {}

    
    def render(self):
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.state = self.S0
        self.env_phase = 0 
        
        return self._get_obs(), {}
        
    def close(self):
        pass

    def set_reward(self, reward):
        pass
          
        
    def get_rewards(self):
        pass
    
    def get_arms(self):
        pass
        
    def get_statistics(self):
        pass
           
    def get_iter(self):
        return self.DINIT + self.DDRUG