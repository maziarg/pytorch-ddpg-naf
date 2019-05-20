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
        self.max_action_limit = max(env.action_space.high)
        self.min_action_limit = min(env.action_space.low)
        self.betta = betta
        self.actor_target_function = actor_target_function
        self.number_of_goal = 1 #Maziar please check! have changed number of goals from 0 to 1 due to division by zero error.
        self.i = 1
        self.g = 0
        self.C_vector = torch.zeros(1, env.observation_space.shape[0])
        self.delta_g = 0
        self.b = 0
        self.B_vector = torch.zeros(1, env.observation_space.shape[0]) #Maziar please check! what is its shape? I am not sure if this is correct!
        self.C_theta = 0
        self.L = -1
        self.U = 1
        self.t = 0  # to account for time step
        self.w_old = torch.zeros(1, env.observation_space.shape[0])
        self.w_new = torch.zeros(1, env.observation_space.shape[0])
        self.eta = None

    def select_action(self, state, previous_action):
        if (self.t == 0):
            self.t += 1
            return torch.FloatTensor(1, 6).uniform_(self.min_action_limit, self.max_action_limit)

        elif ((self.delta_g > self.U) or (self.delta_g < self.L)):
            action = self.actor_target_function(state)
            self.reset_parameters_PolyRL()
            return action

        else:
            self.eta = abs(np.random.normal(self.lambda_, np.sqrt(self.sigma_squared)))
            return self.sample_action_algorithm(previous_action)

    # This function resets parameters of PolyRl every episode. Should be called in the beggining of every episode
    def reset_parameters_in_beginning_of_episode(self):
        self.epsilon = 1 / (1 + self.betta) ** (self.number_of_goal ** 2)
        self.i = 1
        self.g = 0
        self.C_vector = torch.zeros(1, self.env.observation_space.shape[0])
        self.delta_g = 0
        self.b = 0
        self.B_vector = torch.zeros(1, self.env.observation_space.shape[0])
        self.C_theta = 0
        self.L = -1
        self.U = 1
        self.t = 0  # to account for time step
        self.w_old = torch.zeros(1, self.env.observation_space.shape[0])
        self.w_new = torch.zeros(1, self.env.observation_space.shape[0])
        self.eta = None

    def sample_action_algorithm(self, previous_action):
        P = torch.FloatTensor(previous_action.shape[1]).uniform_(self.min_action_limit, self.max_action_limit)
        D = torch.dot(P, previous_action.reshape(-1)).item()
        norm_previous_action = np.linalg.norm(previous_action.numpy(), ord=2)
        V_p = (D / norm_previous_action ** 2) * previous_action
        V_r = P - V_p
        l = np.linalg.norm(V_p.numpy(), ord=2) * np.tan(self.eta)
        k = l / np.linalg.norm(V_r.numpy(), ord=2)
        Q = k * V_r + V_p
        if (D > 0):
            action = Q
        else:
            action = -Q
        action = np.clip(action.numpy(), self.min_action_limit, self.max_action_limit)
        self.i += 1
        return torch.from_numpy(action).reshape(1, action.shape[1])

    def update_parameters(self, previous_state, new_state):
        self.w_old = self.w_new
        norm_w_old = np.linalg.norm(self.w_old.numpy(), ord=2)
        self.w_new = new_state - previous_state
        norm_w_new = np.linalg.norm(self.w_new.numpy(), ord=2)
        self.B_vector = self.B_vector + self.i * self.w_new
        if (self.i != 1):
            Delta1 = previous_state - self.C_vector
            self.old_g = self.g
            self.g = (1 / self.i) * self.g + (1 / self.i) * np.linalg.norm(Delta1.numpy(), ord=2) ** 2
            self.C_theta = ((self.i - 2) * self.C_theta + torch.dot(self.w_new.reshape(-1),
                                                                    self.w_old.reshape(-1)).item() / (
                                    norm_w_new * norm_w_old)) / (self.i - 1)
            Lp = 1 / abs(np.log(self.C_theta)) #Maziar please check! Sometimes gives invalid value for log!
            K = 0
            for j in range(1, self.i):
                K = K + j * np.exp((j - self.i) / Lp)
            norm_B_vector = np.linalg.norm(self.B_vector.numpy(), ord=2)
            last_term = (1 / (self.i - 1)) * self.old_g

            #Upper bound and lower bound are computed here
            self.U = (1 / ((self.i ** 3) * (1 - self.epsilon))) * (
                    (self.i ** 2) * self.b + (norm_B_vector ** 2) + 2 * self.i * self.b * K)-last_term

            self.L = (1 - np.sqrt(2 * self.epsilon)) * (
                    self.b / self.i + (((self.i - 1) * (self.i - 2)) / self.i ** 2) * self.b * np.exp(
                (-abs(self.i - 1))/Lp)+(1/self.i**3)*norm_B_vector**2)-last_term

            self.L=max(0,self.L)

        self.b=((self.i-1)*self.b+norm_w_new**2)/self.i
        self.C_vector=((self.i-1)*self.C_vector+new_state)/self.i
        self.t+=1

    # This function resets the parameters of class
    def reset_parameters_PolyRL(self):
        self.i = 1
        self.g = 0
        self.C_vector = np.zeros(self.env.observation_space.shape[0])
        self.delta_g = 0
        self.b = 0
        self.B_vector = np.zeros(self.env.observation_space.shape[0])
        self.C_theta = 0
        self.L = -1
        self.U = 1
        self.t = 0
        self.w_new = np.zeros(self.env.observation_space.shape[0])
        self.w_old = np.zeros(self.env.observation_space.shape[0])
