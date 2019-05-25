import argparse
from tensorboardX import SummaryWriter
import datetime
import time
import logging
import pickle

logger = logging.getLogger(__name__)
import sys

sys.path.insert(0,"/home/hossain_aboutalebi_gmail_com/pytorch-ddpg-naf")

# Important Note: for running on server you should specify the python project directory ...
# ...the project may contain multiple directories. This should be set manually.
# sys.path.insert(0, '/home/hossain_aboutalebi_gmail_com/pytorch-ddpg-naf')

import gym
import roboschool
import numpy as np
import torch
import os

from policy_engine.mod_reward import *
from policy_engine.poly_rl import *
from policy_engine.ddpg import DDPG
from policy_engine.naf import NAF
from policy_engine.normalized_actions import NormalizedActions
from policy_engine.ounoise import OUNoise
from policy_engine.replay_memory import ReplayMemory, Transition

parser = argparse.ArgumentParser(description='PyTorch poly Rl exploration implementation')

# MAX_PATH_LEN =  20000 # max length of an episode. TODO: add this feature to episode if needed


# *********************************** Environment Setting ********************************************

parser.add_argument('--algo', default='DDPG',
                    help='algorithm to use: DDPG | NAF')

parser.add_argument('-o', '--output_path', default=os.path.expanduser('~') + '/results_exploration_policy',
                    help='output path for files produced by the agent')

parser.add_argument('--sparse_reward', action='store_false',
                    help='for making reward sparse. Default=True')

parser.add_argument('--env_name', default="RoboschoolReacher-v1",
                    help='name of the environment to run')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--threshold_sparcity', type=float, default=1, metavar='G',
                    help='threshold_sparcity for rewards (default: 0.15)')

parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')

parser.add_argument('--reward_negative', action='store_false',
                    help='Determines if we can have neagative reward (Default True)')

parser.add_argument('--num_steps', type=int, default=20, metavar='N',
                    help='max episode length (default: 20)')

parser.add_argument('--num_episodes', type=int, default=1500, metavar='N',
                    help='number of episodes (default: 1500)')

parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')

# Important: batch size here is different in semantics
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128). This is different from usual deep learning work'
                         'where batch size infers parallel processing. Here, we currently do not have that as'
                         'we update our parameters sequentially. Here, batch_size means minimum length that '
                         'memory replay should have before strating to update model parameters')

# *********************************** DDPG Setting ********************************************

# This is the factor for updating the target policy with delay based on behavioural policy
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')

parser.add_argument('--lr_actor', type=float, default=1e-3,
                    help='learning rate for actor policy')

parser.add_argument('--lr_critic', type=float, default=1e-3,
                    help='learning rate for critic policy')

# Note: The following noise are for the behavioural policy of the pure DDPG without the poly_rl policy
parser.add_argument('--ou_noise', type=bool, default=True,
                    help="This is the noise used for the pure version DDPG (without poly_rl_exploration)"
                         " where the behavioural policy has perturbation in only mean of target policy")

parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')

parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')

parser.add_argument('--poly_rl_exploration_flag', action='store_false',
                    help='for using poly_rl exploration. Default=True')

parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')

parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')

# very important factor. Should be investigated in the future.
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')

# *********************************** Poly_Rl Setting ********************************************

parser.add_argument('--betta', type=float, default=0.0001)

parser.add_argument('--epsilon', type=float, default=0.999)

parser.add_argument('--sigma_squared', type=float, default=0.00007)

parser.add_argument('--lambda_', type=float, default=0.035)

# retrieve arguments set by the user
args = parser.parse_args()

# configuring logging
file_path_results = args.output_path + "/" + str(datetime.datetime.now()).replace(" ", "_")
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
os.mkdir(file_path_results)
logging.basicConfig(level=logging.INFO, filename=file_path_results + "/log.txt")
logging.getLogger().addHandler(logging.StreamHandler())

logger.info("=================================================================================")
Config_exeriment = "\n Experiment Configuration:\n*Algorithm: " + str(args.algo) + "\n*Output_path result: " + \
                   str(args.output_path) + "\n*sparse_reward: " + str(
    args.sparse_reward) + "\n*Environment Name: " + str(
    args.env_name) + \
                   "\n*Gamma " + str(args.gamma) + "\n*Max episode steps length: " + str(
    args.num_steps) + "\n*Number of episodes: " \
                   + str(args.num_episodes) + "\n*Tau: " + str(args.tau) + "\n*Learning rate of critic net: " + str(
    args.lr_critic) + "\n*Learning rate of actor net: " \
                   + str(args.lr_actor) + "\n*PolyRL flag: " + str(
    args.poly_rl_exploration_flag) + "\n*Betta of PolyRL: " + str(args.betta) \
                   + "\n*Epsilon of PolyRL: " + str(args.epsilon) + "\n*sigma_squared of PolyRL: " + str(
    args.sigma_squared) + "\n*Lambda of PolyRL: " + str(
    args.lambda_) + "\n*Threshold of sparcity start in rewards: " + str(args.threshold_sparcity) + \
                   "\n*Reward negative has been activated: " + str(args.reward_negative)
logger.info(Config_exeriment)
logger.info("=================================================================================")

env = gym.make(args.env_name)

# for tensorboard
try:
    writer = SummaryWriter(logdir=file_path_results)
except:
    writer = SummaryWriter(file_path_results)

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
                 num_inputs=env.observation_space.shape[0], action_space=env.action_space,
                 lr_actor=args.lr_actor, lr_critic=args.lr_critic)

poly_rl_alg = None
if (args.poly_rl_exploration_flag):
    poly_rl_alg = PolyRL(gamma=args.gamma, betta=args.betta, epsilon=args.epsilon, sigma_squared=args.sigma_squared,
                         actor_target_function=agent.select_action_from_target_actor, env=env, lambda_=args.lambda_)
    agent.set_poly_rl_alg(poly_rl_alg)

# Important Note: This replay memory shares memory with different episodes
memory = ReplayMemory(args.replay_size)

# Adds noise to the selected action by the policy"
ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None

rewards = []
total_numsteps_episode = 0
total_numsteps = 0
updates = 0
Final_results = {"reward": [],"modified_reward":[]}
start_time = time.time()
for i_episode in range(args.num_episodes):
    total_numsteps_episode = 0
    state = torch.Tensor([env.reset()])

    if args.ou_noise:
        # I did not change the implementation of ounoise from the source! (for pure DDPG without poly_rl exploration)
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if (args.poly_rl_exploration_flag):
        poly_rl_alg.reset_parameters_in_beginning_of_episode(i_episode + 2)

    episode_reward = 0
    previous_action = None
    previous_state = state
    counter = 0
    while (counter < args.num_steps):
        total_numsteps += 1
        action = agent.select_action(state=state, action_noise=ounoise, previous_action=previous_action, tensor_board_writer=writer,
                                     step_number=total_numsteps)
        previous_action = action
        next_state, reward, done, info_ = env.step(action.cpu().numpy()[0])
        total_numsteps_episode += 1
        episode_reward += reward
        action = torch.Tensor(action.cpu())
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        modified_reward,flag_absorbing_state = make_reward_sparse(env=env, flag_sparse=args.sparse_reward, reward=reward,
                                             threshold_sparcity=args.threshold_sparcity, negative_reward_flag=args.reward_negative,num_steps=args.num_steps)
        modified_reward = torch.Tensor([modified_reward])
        memory.push(state, action, mask, next_state, modified_reward)
        previous_state = state
        state = next_state
        if (args.poly_rl_exploration_flag and poly_rl_alg.Update_variable):
            poly_rl_alg.update_parameters(previous_state=previous_state, new_state=state, tensor_board_writer=writer)

        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_parameters(batch, tensor_board_writer=writer,
                                                                  episode_number=i_episode)
                updates += 1
        # if the environemnt should be reset, we break
        if done or flag_absorbing_state:
            break
        counter += 1

    writer.add_scalar('reward/train', episode_reward, i_episode)
    rewards.append(episode_reward)

    # This section is for computing the target policy perfomance
    # The environment is reset every 10 episodes automatically and we compute the target policy reward.
    if i_episode % 10 == 0:
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        episode_modified_reward = 0
        counter = 0
        while (counter < args.num_steps):
            action = agent.select_action_from_target_actor(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_reward += reward
            modified_reward, flag_absorbing_state = make_reward_sparse(env=env, flag_sparse=args.sparse_reward, reward=reward,
                                                                       threshold_sparcity=args.threshold_sparcity,
                                                                       negative_reward_flag=args.reward_negative, num_steps=args.num_steps)

            episode_modified_reward+=modified_reward
            next_state = torch.Tensor([next_state])
            state = next_state
            if done or flag_absorbing_state:
                break
            counter += 1

        writer.add_scalar('real_reward/test', episode_reward, i_episode)
        writer.add_scalar('reward_modified/test', episode_modified_reward, i_episode)
        time_len = time.time() - start_time
        start_time = time.time()
        rewards.append(episode_reward)
        Final_results["reward"].append(episode_reward)
        Final_results["modified_reward"].append(episode_modified_reward)
        # last_x_body = env.env.body_xyz[0]
        # writer.add_scalar('x_body', last_x_body, i_episode)
        logger.info(
            "Episode: {}, time:{}, numsteps in the episode: {}, total steps so far: {}, reward: {}, modified_reward {}".format(
                i_episode, time_len, total_numsteps_episode, total_numsteps, episode_reward, episode_modified_reward))

    with open(file_path_results + '/result_reward.pkl', 'wb') as handle:
        pickle.dump(Final_results, handle)
env.close()
