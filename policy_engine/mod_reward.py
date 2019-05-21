
import torch

def make_reward_sparse(env,reward,flag_sparse,threshold_sparcity):
    if flag_sparse is True:
        if (env.env.body_xyz[0] > threshold_sparcity):
            reward = torch.Tensor([1])
        else:
            reward = torch.Tensor([0])
    return reward
