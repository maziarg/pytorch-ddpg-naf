
import torch

def make_reward_sparse(env,reward,flag_sparse,threshold_sparcity,negative_reward_flag,num_steps):
    flag_absorbing_state=False
    if flag_sparse is True:
        if (reward > threshold_sparcity):
            reward = reward
            # flag_absorbing_state=True
        else:
            reward = 0
    return reward,flag_absorbing_state
