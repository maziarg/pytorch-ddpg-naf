#import matplotlib.pyplot as plt
#from PIL import Image
#%matplotlib inline
#from IPython import display

#%load_ext autoreload
#%autoreload 2
import math
import gym
import numpy as np
import sys
from collections import namedtuple
import argparse
import datetime
import os
import time
import shutil
from itertools import chain
#import dill
import random
import numpy as np
import random

import torch
import matplotlib.pyplot as plt

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import csv

from tensorboardX import SummaryWriter
#---- Global params

SEED = 123 #seed

DISCOUNT_FACTOR = 0.99

#---- exploration method

# the arguments for the vis agent here
LAMBDA = np.linspace(0.01,0.5,10)[3]
print(LAMBDA)

# kind of noise to use
# NOISE_METHODS = "traj_epsilon" # PolyRL with increasing exploitation over time
NOISE_METHODS = "traj" # polyRL with radius of gyration
# NOISE_METHODS = "ou" # OU
# NOISE_METHODS = None # uniform sampling

# The parameters needed if "traj_epsilon" is chosen as noise_method
EPSILON = 1 # The probability of exploration vs exploitation
ALPHA = 0.001 # Defines the rate of the decrease in EPSILON

POLY_STD_DEV = 0.2 # Standard deviation for PolyRL
OU_STD_DEV = 3 # Standard deviation for OU_Noise


#----- state feature mapping variables
N_FEAT_SAMPLES = 10000
N_FEAT_COMPONENTS = 100


#---- learning params
UPDATES_PER_STEP = 1
BATCH_SIZE = 128
REPLAY_SIZE = 10000
LR = 1e-1


#---- experiment book-keeping params
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 10000
GPU = False
OUT = '/Users/susanamin/Documents/Poly RL/Harsh Code/LinearFunctionApproximation/Susan_version/models'
LOG_DIR = '/Users/susanamin/Documents/Poly RL/Harsh Code/LinearFunctionApproximation/Susan_version/logs'
RESET_DIR = True
#EVAL_EVERY = 1000 (Susan: removed in the new version)
EVAL_N = 10
EVAL_MAX_PATH_LEN = 10000
TAU = 0.001

# Used for Tile coding
NUM_TILES_STATE = 900
NUM_TILES_ACTION = 10

# Susan: Removed the following lines
#random.seed(SEED)
#torch.manual_seed(SEED)
#np.random.seed(SEED)

# create the env here

#from envs.continous_grids import GridWorld

env = GridWorld(max_episode_len = MAX_PATH_LEN, num_rooms=1, Graph=graph)
# env = gym.envs.make("MountainCar-v0")
# create feature kernel

# Feature Preprocessing: Normalize to zero mean and unit variance
# Use samples from the observation space to do this
#observation_examples = np.array([env.observation_space.sample() for x in range(N_FEAT_SAMPLES)])
#scaler = sklearn.preprocessing.StandardScaler()
#scaler.fit(observation_examples)
#
# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
#featurizer = sklearn.pipeline.FeatureUnion([
#        ("rbf1", RBFSampler(gamma=5.0, n_components=N_FEAT_COMPONENTS)),
#        ("rbf2", RBFSampler(gamma=2.0, n_components=N_FEAT_COMPONENTS)),
#        ("rbf3", RBFSampler(gamma=1.0, n_components=N_FEAT_COMPONENTS)),
#        ("rbf4", RBFSampler(gamma=0.5, n_components=N_FEAT_COMPONENTS))
#        ])
#featurizer = RBFSampler(gamma = 1.0, n_components = N_FEAT_COMPONENTS)


#featurizer.fit(scaler.transform(observation_examples))
# featurize the state

#s = env.observation_space.sample()

# normalize
#ns = scaler.transform([s])



# rbf features
#phi_s = featurizer.transform(ns)


# create the feature mapping function here

#def featurize_state(state):
#    """
#    Returns the featurized representation for a state.
#    """
#    scaled = scaler.transform([state])
#    featurized = featurizer.transform(scaled)
#    return featurized[0]
# create the exploration policy here

# Truncate a float number to "n" decimal places
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

# Grid-Like Tile Coding
tile_size_state = truncate((np.sqrt(env.grid_len ** 2 / NUM_TILES_STATE)), 2)
tile_max_index_state =  math.floor((env.max_position - env.min_position)/tile_size_state)-1
tile_size_action = truncate((2 * math.pi / NUM_TILES_ACTION), 2)
tile_max_index_action = NUM_TILES_ACTION -1 

# def featurize (state, action):
    # state_index = featurize_state (state)
    # phi_a = featurize_action (action)
    # phi_sa = np.concatenate([phi_s, phi_a])
    #return state_index, phi_a

def featurize_state(state):
    # phi_s = np.zeros (NUM_TILES_STATE)
    column_index = math.floor((state[0] - env.min_position)/tile_size_state)
    #Taking care of the last column
    if column_index > tile_max_index_state:
        column_index -= 1
    row_index = math.floor((state[1] - env.min_position)/tile_size_state)
    # Taking care of the last row
    if row_index > tile_max_index_state:
        row_index -= 1
    
    index = (row_index * (tile_max_index_state + 1)) + column_index 
    # phi_s[index] = 1
    return index

def featurize_action(action):
    phi_a = np.zeros(NUM_TILES_ACTION)
    angle = np.arctan2(action[1],action[0])
    if angle < 0:
        angle = 2 * math.pi + angle
    action_index = math.floor(angle/tile_size_action)
    if action_index > tile_max_index_action:
        action_index -= 1
    phi_a[action_index] = 1         
    return action_index, phi_a



nb_actions = env.action_space.shape[0]
max_action_limit = env.action_space.high[0]

# state_dim = featurize_state(env.observation_space.sample()).shape[0]
state_feature_dim = NUM_TILES_STATE
state_dim = env.observation_space.sample().shape[0]

if NOISE_METHODS=="traj":
    # first create OU noise to jumpstart
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(OU_STD_DEV) * np.ones(nb_actions))

    # create OU noise
    action_noise = GyroPolyNoiseActionTraj(lambd = float(LAMBDA),
                                           action_dim = nb_actions,
                                           state_dim = state_dim,
                                           ou_noise = ou_noise,
                                           sigma = float(POLY_STD_DEV),
                                           max_action_limit = max_action_limit)
elif NOISE_METHODS == "traj_epsilon":
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(OU_STD_DEV) * np.ones(nb_actions))
    action_noise = PolyNoiseTrajEpsilon(lambd = float(LAMBDA),
                                        action_dim = nb_actions,
                                        state_dim = state_feature_dim,
                                        ou_noise = ou_noise,
                                        sigma = float(POLY_STD_DEV),
                                        max_action_limit = max_action_limit,
                                        epsilon = EPSILON,
                                        alpha = ALPHA)

else:
    #OU
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(OU_STD_DEV) * np.ones(nb_actions))
d_args = {
    "env" : env,
    "Graph": graph,
    #"num_steps" : NUM_STEPS, (Susan: Removed in the new version)
    "num_ep":NUM_EP, #Susan: Added in the new version,
    "num_iter": NUM_ITER, # (Susan: added in the new version)
    "max_path_len": MAX_PATH_LEN,
    "seed":SEED,
    "gpu" : False,
    "replay_size": REPLAY_SIZE,
    "gamma": DISCOUNT_FACTOR,
    "updates_per_step" : UPDATES_PER_STEP,
    "batch_size" : BATCH_SIZE,
    "lr" : LR,
    "log_interval" : LOG_INTERVAL,
    "checkpoint_interval" : CHECKPOINT_INTERVAL,
    "gpu" : GPU,
    "out" : OUT,
    "log_dir" : LOG_DIR,
    "reset_dir" : RESET_DIR,
    #"eval_every" : EVAL_EVERY, # Susan: Removed in the new verison
    "eval_n" : EVAL_N,
    "eval_max_path_len" : EVAL_MAX_PATH_LEN,
    "tau" : TAU,
    "state_num_features" : NUM_TILES_STATE,
    "action_num_features" : NUM_TILES_ACTION,

}


# create files for logs and check-pointing

# create the dire
toprint = ['max_path_len', 'num_ep','eval_max_path_len', 'lr']

name = ''
for arg in toprint:
    name += '_{}_{}'.format(arg, d_args[arg])

# more name parameters
if NOISE_METHODS == "traj": # Susan: added the if statement because of STD_DEV
    name += "_" + str(NOISE_METHODS) + "_" + str(POLY_STD_DEV) + "_" + str(LAMBDA)
elif NOISE_METHODS == "traj_epsilon":
    name += "_" + str(NOISE_METHODS) + "_" + str(POLY_STD_DEV) + "_" + str(LAMBDA) + "_" + str(ALPHA)
else:
    name += "_" + str(NOISE_METHODS) + "_" + str(OU_STD_DEV) + "_" + str(LAMBDA)

out_dir = os.path.join(d_args["out"], "2DGrid", name)
d_args["out"] = out_dir


# convert everything to agent input type

from collections import namedtuple

args = namedtuple("parser", d_args.keys())(*d_args.values())
args.out
# create the directory here and tensorboard writer here

os.makedirs(args.out, exist_ok=True)

# create the tensorboard summary writer here
tb_log_dir = os.path.join(args.log_dir, "2DGrid", name, 'tb_logs')

#Susan added this line
csv_log_dir = os.path.join(args.log_dir, "2DGrid", name, 'csv_logs')

print("Log dir", tb_log_dir)
print("Out dir", args.out)
#Susan added this line
print("csv log dir", csv_log_dir)

if args.reset_dir:
    shutil.rmtree(tb_log_dir, ignore_errors=True)
    #Susan added this line
    shutil.rmtree(csv_log_dir, ignore_errors=True)
os.makedirs(tb_log_dir, exist_ok=True)
#Susan added this line
os.makedirs(csv_log_dir, exist_ok=True)
tb_writer = SummaryWriter(log_dir=tb_log_dir)

agent = LinearAgent(args, env, action_noise, featurize_state, featurize_action, tb_log_dir, csv_log_dir)
# run the agent here

agent.run()
