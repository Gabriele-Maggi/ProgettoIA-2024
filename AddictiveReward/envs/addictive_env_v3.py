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
        self.DINIT = 50
        self.DDRUG = 100000

        ############ Bandit ##############
        self.arms = 2
                
        self.number_action = np.ones(self.arms)
        self.reward_action = np.ones(self.arms)
        self.q_bandit = np.zeros(self.arms)
        
        self.t = 1 # aggiornato a ogni azione presa da agent
        self.c = 1.5
        self.current_arm = 0 # 0:4 1:2
        self.non_addictive_reward = -1
        ##################################
        
    def get_iter(self):
        return self.DINIT + self.DDRUG
        
    def _get_obs(self):
        return self.state
    
    ## bandit method ##
    def _calculate_rew(self):
        for ar in range(self.arms):
            self.q_bandit[ar] = self.reward_action[ar] / self.number_action[ar] 
    
    def _calculate_action(self):
        self._calculate_rew()
        
        temp = np.zeros(self.arms)
        for ar in range(self.arms):
            temp[ar] = (self.q_bandit[ar] + (self.c * math.sqrt(math.log(self.t) / self.number_action[ar] )))
        #print(f"{temp}")    
        new_arm = np.argmax(temp)
        if self.current_arm != new_arm:
            self.t += 1
        self.current_arm = new_arm
                        
        
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

                self.reward_action[self.current_arm] -= 1
                self._calculate_action()
                
            elif action == 4:
                
                self.reward_action[self.current_arm] += 1
                self.number_action[self.current_arm] += 1
                
                self._calculate_action()
                
                #print(f"{self.reward_action}")
                
                if self.current_arm == 0:
                    self.state = 4
                    reward = self.reward_addicted       
                elif self.current_arm == 1:
                    self.state = 2
                    reward = self.non_addictive_reward 
            else:
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
        
        return self._get_obs(), reward, terminated, False, {}

    
    def render(self):
        pass

    def reset_bandit(self):
        self.number_action = np.ones(self.arms)
        self.reward_action = np.ones(self.arms)
        self.q_bandit = np.zeros(self.arms)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)
        self.t = 1 # aggiornato a ogni azione presa da agent
        self.state = self.S0
        self.env_phase = 0 
        
        return self._get_obs(), {}
        
    def close(self):
        pass 
        

