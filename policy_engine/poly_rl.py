import os
import shutil
import numpy as np
import torch


# implements the PolyRL algorithm
# It is implemented based on psudo code of algorithm
class PolyRL():

    def __init__(self, gamma, env, lambda_=0.08, betta=0.001, epsilon=0.999, sigma_squared=0.04,
                 actor_target_function=None):
        self.epsilon = epsilon
        self.env = env
        self.gamma = gamma
        self.lambda_ = lambda_
        self.sigma_squared = sigma_squared
        self.nb_actions = env.action_space.shape[0]
        self.max_action_limit = max(env.action_space)
        self.min_action_limit = min(env.action_space.low)
        self.betta = betta
        self.actor_target_function = actor_target_function
        self.number_of_goal = 0
        self.number_of_timed_poly_rl_is_played = 1
        self.g = 0
        self.C_vector = np.zeros(env.observation_space.shape[0])
        self.delta_g = 0
        self.b_vector = np.zeros(env.action_space.shape[0])
        self.B_vector = np.zeros(env.action_space.shape[0])
        self.C_theta = 0
        self.L = -1
        self.U = 1
        self.t = 0  # to account for time step
        self.w_old = np.zeros(env.observation_space.shape[0])
        self.w_new = np.zeros(env.observation_space.shape[0])
        self.eta=None

    def select_action(self, state,previous_action):
        if (self.t == 0):
            return torch.FloatTensor(1, 6).uniform_(self.min_action_limit, self.max_action_limit)

        elif ((self.delta_g > self.U) or (self.delta_g < self.L)):
            action = self.actor_target_function(state)
            self.reset_parameters_PolyRL()
            return action

        else:
            self.eta=abs(np.random.normal(self.lambda_,np.sqrt(self.sigma_squared)))
            self.sample_action_algorithm(previous_action)

    def sample_action_algorithm(self,previous_action):
        P=torch.FloatTensor(previous_action.shape[1]).uniform_(self.min_action_limit, self.max_action_limit)
        D=torch.dot(P,previous_action.reshape(-1)).item()
        norm_previous_action=np.linalg.norm(previous_action.numpy(),ord=2)
        V_p=(D/norm_previous_action)*previous_action
        V_r=P-V_p
        l=np.linalg.norm(V_p.numpy(),ord=2)*np.tan(self.eta)
        k=l/np.linalg.norm(V_r.numpy(),ord=2)
        Q=k*V_r+V_p
        if(D>0):
            action=Q
        else:
            action=-Q
        action=np.clip(action.numpy(),self.min_action_limit,self.max_action_limit)
        return torch.from_numpy(action).reshape(1,action.shape[1])

    #This function resets the parameters of class
    def reset_parameters_PolyRL(self):
        self.number_of_timed_poly_rl_is_played = 1
        self.g = 0
        self.C_vector = np.zeros(self.env.observation_space.shape[0])
        self.delta_g = 0
        self.b_vector = np.zeros(self.env.action_space.shape[0])
        self.B_vector = np.zeros(self.env.action_space.shape[0])
        self.C_theta = 0
        self.L = -1
        self.U = 1
        self.t = 0
        self.w_new = np.zeros(self.env.observation_space.shape[0])
        self.w_old = np.zeros(self.env.observation_space.shape[0])

