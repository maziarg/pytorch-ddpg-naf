import argparse
from tensorboardX import SummaryWriter
import sys

# Important Note: for running on server you should specify the python project directory ...
# ...the project may contain multiple directories. This should be set manually.
sys.path.insert(0, '/home/hossain_aboutalebi_gmail_com/pytorch-ddpg-naf')

import gym
import roboschool
import numpy as np
import torch
import os

from policy_engine.ddpg import DDPG
from policy_engine.naf import NAF
from policy_engine.normalized_actions import NormalizedActions
from policy_engine.ounoise import OUNoise
from policy_engine.replay_memory import ReplayMemory, Transition

parser = argparse.ArgumentParser(description='PyTorch poly Rl exploration implementation')

parser.add_argument('--algo', default='NAF',
                    help='algorithm to use: DDPG | NAF')

parser.add_argument('--env-name', default="RoboschoolHalfCheetah-v1",
                    help='name of the environment to run')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')

# Note: The following noise are for the behavioural policy of the pure DDPG without the poly_rl policy
parser.add_argument('--ou_noise', type=bool, default=True,
                    help="This is the noise used for the pure version DDPG (without poly_rl_exploration)"
                         " where the behavioural policy has perturbation in only mean of target policy")

parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')

parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')

parser.add_argument('--poly_rl_exploration_flag', action='store_true',
                    help='for using poly_rl exploration')

parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')

parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')

# Important: batch size here is different in semantics
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128). This is different from usual deep learning work'
                         'where batch size infers parallel processing. Here, we currently do not have that as'
                         'we update our parameters sequentially. Here, batch_size means minimum length that '
                         'memory replay should have before strating to update model parameters')

parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')

parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')

parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')

parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')

# very important factor. Should be investigated in the future.
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')

# retrieve arguments set by the user
args = parser.parse_args()

# Important Note: This NormalizedActions class should be tested on each environment to make sure the ...
# ... way it is normalzes the action is compatible with the environment. It might be better to remove it for now!
env = NormalizedActions(gym.make(args.env_name))

# for tensorboard
writer = SummaryWriter()

# sets the seed for making it comparable with other implementations
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# currently we only support DDPG. TODO: implement other algorithms for future
if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.hidden_size,
                env.observation_space.shape[0], env.action_space)
else:
    agent = DDPG(gamma=args.gamma, tau=args.tau, hidden_size=args.hidden_size,
                 poly_rl_exploration_flag=args.poly_rl_exploration_flag,
                 num_inputs=env.observation_space.shape[0], action_space=env.action_space)

# Important Note: This replay memory shares memory with different episodes
memory = ReplayMemory(args.replay_size)

# Adds noise to the selected action by the policy"
ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None


rewards = []
total_numsteps = 0
updates = 0

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])

    if args.ou_noise:
        # I did not change the implementation of ounoise from the source! (for pure DDPG without poly_rl exploration)
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    episode_reward = 0
    while True:
        action = agent.select_action(state, ounoise)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        total_numsteps += 1
        episode_reward += reward

        action = torch.Tensor(action.cpu())
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        memory.push(state, action, mask, next_state, reward)

        state = next_state

        # If the batch_size is bigger than memory then we do not need memory replay! lol
        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)

                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                updates += 1
        # if the environemnt should be reset, we break
        if done:
            break

    writer.add_scalar('reward/train', episode_reward, i_episode)

    rewards.append(episode_reward)

    #This section is for computing the target policy perfomance
    #The environment is reset every 10 episodes automatically and we compute the target policy reward.
    if i_episode % 10 == 0:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_reward += reward

            next_state = torch.Tensor([next_state])
            state = next_state
            if done:
                break

        writer.add_scalar('reward/test', episode_reward, i_episode)

        rewards.append(episode_reward)
        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps,
                                                                                       rewards[-1],
                                                                                       np.mean(rewards[-10:])))


env.close()
