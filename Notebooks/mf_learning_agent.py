import numpy as np
import random

class MFLearningAgent:
    def __init__(self, 
                 learning_rate ,
                 initial_epsilon,
                 epsilon_decay, 
                 final_epsilon,
                 discount_factor ,
                 obs_space_env, 
                 act_space_env,):
        
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.q_values = np.zeros([obs_space_env.n, act_space_env.n])

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.training_error = []

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon: # epsilon greedy 
            return act_space_env # Explore action space
        else:
            return np.argmax(self.q_values[state]) # Exploit learned values
    
    # def update_value_iteration(self, state, action, reward, next_state):
    #     old_value = self.q_values[state, action] 
    #     next_max = np.max(self.value_iteration[next_state, :])
    #     self.q_values[state, action] =  (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max) 

    def update(self, obs: int, action: int, reward: float, terminated: bool, next_obs: int):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

        
    def decay_epsilon(self):
       self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



