import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium import spaces
import random
from numpy.random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional


class AddictiveEnv_v1(gym.Env):
    def __init__(self):
        # PARAMS
        self.NT = 23  # Numero di stati + 1 
        self.NA = 9   # Numero of actions
        self.S0 = 4   # Starting state
        self.RP = -4  # Punishment end of Addictive Area
        self.RC = -1.2  # Punishment in Addictive Area
        self.RDD = 10  # Reward at entering Addictive reward state
        self.RG = 1    # Reward 
        
        self.DINIT = 5 # Duration safe phase
        self.DDRUG = 950

        self.pmm = 0.4
        self.pm = 0.8
        
        ## ACTIONS
        #as2-7 -> 0-5
        self.AG = 6
        self.AW = 7
        self.AD = 8

    
        
        self.AR = ["as2", "as3", "as4", "as5", "as6", "as7", "aG", "aW", "aD"]
        # ag -> 6
        # aw -> 7
        # ad -> 8
        # Observation space (20 stati 1 healty, 2-7 neutral 8-22 dipendenza)
        self.observation_space = spaces.Discrete(self.NT) # non uso stato 0 
        # Actions spaces (9 azioni -> as 2-7, ag, aw, ad)
        self.action_space = spaces.Discrete(self.NA) 
        # stato iniziale: 4        
        self.state = self.S0
        # env_phase: 0-50 safe -> addictive deactivated
        # env_phase: 50-2000 safe -> addictive activated
        self.env_phase = 0 
           

    def get_iter(self):
        return self.DINIT + self.DDRUG

    def _get_obs(self):
        return self.state

    
    def step(self, action):
        terminated = False
        info = {}
        # safe phase
        if (self.env_phase < self.DINIT):
            self.action_space = spaces.Discrete(self.NA - 1)
        else:
            self.action_space = spaces.Discrete(self.NA)

        
        if (self.state < 8):
            if (action < 6 and self.state != 1):
                 if (self.state - 1 <= action + 2 <= self.state + 1):
                    info = "mi muovo da " + str(self.state) + " a " + str(action + 2)+ " action: " + self.AR[action] + " 6"
                    self.state = action + 2
                    reward = 0
                     
                 else:
                    s = str(self.state)
                    sampleList = [self.state, action + 2] # con probabilità p << 1 il salto di stato, se no rimango dove sono 
                    self.state = choice(sampleList, 1, p = [(1-self.pmm), self.pmm])[0]
                    info = "mi muovo da " + s + " a " + str(self.state)+ " action: " + self.AR[action] + " 7"
                    reward = 0
                     
            elif (action == 6 and self.state == 2):
                info = "mi muovo da " + str(self.state) + " a healty" + " action: " + self.AR[action] + " 8"
                self.state = 1
                reward = 0
                
            elif (action == 6 and self.state == 1): # stato sano
                info = "mi muovo da " + "healty" + " a " + str(self.S0) + " action: " + self.AR[action] + " 9"
                self.state = self.S0
                reward = self.RG
                
            elif (action == self.AD and self.state == 7): # stato dipendenza
                self.state = 8
                info = "addicted"+ " action: " + self.AR[action] + " 10"
                reward = self.RDD
                
            else:
                self.state = self.state
                info = "non mi muovo da " + str(self.state) + " 11"
                reward = 0
                
        elif (self.state >= 8 and self.state <= 22): # stati di after effects
            if (self.state == 20 and (action == self.AW or action == self.AD)):
                s = str(self.state)
                sampleList = [self.state, self.S0] # con probabilità p << 1 il salto di stato, se no rimango dove sono 
                self.state = choice(sampleList, 1, p = [(1-self.pmm), self.pmm])[0]
                info = "mi muovo da " + s + " a " + str(self.state) + " action: " + self.AR[action] + " 12"
                if(self.state == self.S0):
                    reward = self.RP
                else:
                    reward = self.RC
                
            elif (self.state == 15 and action == self.AW):
                s = str(self.state)
                sampleList = [self.state, self.S0] # con probabilità p < 1 il salto di stato, se no rimango dove sono 
                self.state = choice(sampleList, 1, p = [(1-self.pm), self.pm])[0]
                info = "mi muovo da " + s + " a " + str(self.state) + " action: " + self.AR[action] + " 13"
                reward = self.RP
                
            elif (action == self.AW or action == self.AD):
                if (self.state != 8 and self.state != 22):
                    s = str(self.state)
                    sampleList = [self.state, self.state - 1, self.state + 1] # con probabilità p < 1 il salto di stato, se no rimango dove sono 
                    self.state = choice(sampleList, 1, p = [(1 - (self.pm)), self.pm/2, self.pm/2])[0]
                    info = "mi muovo da " + s + " a " + str(self.state) + " action: " + self.AR[action] + " 14"
                    reward = self.RC
                elif (self.state == 8):
                    s = str(self.state)
                    sampleList = [self.state, self.NT - 1, self.state + 1] # con probabilità p < 1 il salto di stato, se no rimango dove sono 
                    self.state = choice(sampleList, 1, p = [(1 - (self.pm)), self.pm/2, self.pm/2])[0]
                    info = "mi muovo da " + s + " a " + str(self.state) + " action: " + self.AR[action] + " 15"
                    reward = self.RC
                elif (self.state == 22):
                    s = str(self.state)
                    sampleList = [self.state, self.state - 1, 8] # con probabilità p < 1 il salto di stato, se no rimango dove sono 
                    self.state = choice(sampleList, 1, p = [(1 - (self.pm)), self.pm/2, self.pm/2])[0]
                    info = "mi muovo da " + s + " a " + str(self.state) + " action: " + self.AR[action] + " 16"
                    reward = self.RC
            else:
                info = "non mi muovo da " + str(self.state) + " 17"
                reward = -1.2
                

        # check if done
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
       pass
        
