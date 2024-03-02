import heapq
import numpy as np
import numba as nb
from numba.experimental import jitclass

class MBLearningAgent:
    def __init__(self, 
                 learning_rate: float, 
                 initial_epsilon: float, 
                 epsilon_decay: float, 
                 final_epsilon: float, 
                 obs_space_env,
                 act_space_env,
                 discount_factor: float = 0.9, 
                 mbus = 50 ,
                 theta=0.01, 
                 ):
        
        
        self.q_values = np.zeros([obs_space_env.n, act_space_env.n])

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.mbus = mbus
        self.transition_model = {} 
        self.initialize_transition_model()
        
        self.training_error = []
        
        self.pred_queue = PriorityQueue()
        self.theta = theta 
        self.upd = 0
    
    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon: # epsilon greedy 
            return act_space_env.sample() # Explore action space
        else:
            return np.argmax(self.q_values[state]) # Exploit learned values

    def value_iteration(self, iter):
        epsilon = 0.1
        for _ in range(1, iter):
            delta = 0
            for state in range(1, obs_space_env.n):
                for action in range(0,  act_space_env.n):
                    t = 0
                    for next_state in self.transition_model[state][action]:
                        probability = self.transition_model[state][action][next_state]['probability']
                        reward = self.transition_model[state][action][next_state]['reward']
                        t += probability * (reward + self.discount_factor * np.max(self.q_values[next_state]))
                    self.q_values[state][action] = t
                     
                Vs = abs(self.q_values[state][action] - np.max(self.q_values[state]))
                delta = max(delta, Vs)
                
            if delta < epsilon and delta > 0:
                break

    def prioritized_sweeping(self, state, action):
        self.calculate_sweep(state, action)
        while not (self.pred_queue.is_empty()):
            if self.upd > self.mbus:
                #print("stopped")
                self.pred_queue.clean()
                break
            state = self.pred_queue.pop()
                        
            for state_p in range(1, obs_space_env.n):
                for action_p in range(act_space_env.n): 
                    for next_state_p in self.transition_model[state_p][action_p]:
                        if next_state_p == state:
                            self.calculate_sweep(state_p, action_p)
                            

    def calculate_sweep(self, state, action):
        old_q = self.q_values[state][action]
        new_q = 0
        
        for next_state in self.transition_model[state][action]:
            probability = self.transition_model[state][action][next_state]['probability']
            reward = self.transition_model[state][action][next_state]['reward']
            new_q += probability * (reward + self.discount_factor * np.max(self.q_values[next_state]))
        self.q_values[state][action] = new_q
        p = abs(new_q - old_q)
        if p > 0:
            self.upd += 1
            max_q = np.max(self.q_values[state])
            if old_q == max_q or new_q == max_q:
                 self.pred_queue.insert(state, p)
       
    def initialize_transition_model(self):
        for state in range(1, obs_space_env.n):
            self.transition_model[state] = {}
            for action in range(act_space_env.n):
                self.transition_model[state][action] = {}
                for next_state in range(obs_space_env.n):
                    self.transition_model[state][action][next_state] = {'count': 0, 'probability': 0, 'reward': 0}
                    
    def update_transition_model(self, state, action, next_state, reward):
        self.transition_model[state][action][next_state]['count'] += 1
        self.transition_model[state][action][next_state]['reward'] = reward
        self.calculate_transition_probabilities()
        
        #self.prioritized_sweeping(obs, action)
        
        transition_array = transition_model_to_numpy(self.transition_model, obs_space_env.n, act_space_env.n)
        self.q_values = value_iteration_gpu(self.mbus, obs_space_env.n, act_space_env.n, transition_array, self.discount_factor, self.q_values)
        
        #self.value_iteration(self.mbus)
        
    def calculate_transition_probabilities(self):
        
        for state in range(1, obs_space_env.n):
            for action in range(act_space_env.n):
                total_count = sum(self.transition_model[state][action][next_state]['count'] for next_state in self.transition_model[state][action])
                for next_state in self.transition_model[state][action]:
                    
                    c = self.transition_model[state][action][next_state]['count']
                    if total_count == 0:
                        total_count = 1
                    self.transition_model[state][action][next_state]['probability'] = c / total_count
                    #if  self.transition_model[state][action][next_state]['probability'] != 0:
                    #    print(f"{state} {action} {next_state} { self.transition_model[state][action][next_state]['probability']}")
    
    def decay_epsilon(self):
       self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    




### Funzioni
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def is_empty(self):
        if self._queue  == []:
            return True
        return False
        
    def insert(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def clean(self):
        self._queue = []
        self.index = 0
        
    def pop(self):
        return heapq.heappop(self._queue)[-1]



def transition_model_to_numpy(transition_model, obs_space, action_space):
    transition_array = np.zeros((obs_space, action_space, obs_space, 3))  # Assuming 3 fields: count, probability, reward
    
    for state in range(1, obs_space):
        for action in range(action_space):
            for next_state in range(1, obs_space):
                transition_info = transition_model[state][action].get(next_state, {'count': 0, 'probability': 0, 'reward': 0})
                transition_array[state, action, next_state] = [transition_info.get('count', 0), transition_info.get('probability', 0), transition_info.get('reward', 0)]

    #print(transition_array)
    return transition_array



@nb.jit
def value_iteration_gpu(iterations, obs_space, action_space, transition_model, discount_factor, q_values):
    epsilon = 0.1
    delta = 1.0  # Set an initial value for delta
    
    while iterations > 0 and delta > epsilon:
        delta = 0
        
        for state in range(1, obs_space):
            for action in range(action_space):
                t = 0
                for next_state in range(1, obs_space):
                    count, probability, reward = transition_model[state, action, next_state]
                    t += probability * (reward + discount_factor * np.max(q_values[next_state]))
                q_values[state, action] = t
            
                # Calculate delta
                Vs = abs(q_values[state, action] - np.max(q_values[state]))
                delta = max(delta, Vs)
        
        iterations -= 1  # Decrease the number of iterations
    
    return q_values


