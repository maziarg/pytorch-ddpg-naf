

def make_reward_sparse(env,reward,flag_sparse):
    if flag_sparse is True:
        body_dist = env.get_body_com("torso")[0]
        if abs(body_dist) <= 5.0:
            reward = 0.
        else:
            reward = 1.0
        return reward
    else:
        return reward


#for reward sparsity TODO: check if this sparsity makes sense
        # if(args.sparse_reward):
        #     if (env.env.body_xyz[0]>5):
        #         modified_reward=torch.Tensor([1])
        #     else:
        #         modified_reward = torch.Tensor([0])
        # else:
        #     modified_reward=reward