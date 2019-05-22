
import torch

def make_reward_sparse(env,reward,flag_sparse,threshold_sparcity,negative_reward_flag,num_steps):
    flag_absorbing_state=False
    if flag_sparse is True:
        if (env.env.body_xyz[0] > threshold_sparcity):
            reward = torch.Tensor([1])
            flag_absorbing_state=True
        elif(negative_reward_flag):
            reward = torch.Tensor([-1/num_steps])
        else:
            reward = torch.Tensor([0])
    return reward,flag_absorbing_state
