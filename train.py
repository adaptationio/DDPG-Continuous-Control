from unityagents import UnityEnvironment
import numpy as np



import random
import time  
import copy
from collections import namedtuple, deque


import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
#%matplotlib inline

env = UnityEnvironment(file_name='Reacher_Linux_NoVis-V20/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

from agent import Agent

agent = Agent(state_size=state_size, action_size=action_size, random_seed=4)


def ddpg(n_episodes=2000, max_steps=1000):
    scores_hundred = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        highest_score = 0
        average_score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations            
        scores_agents = np.zeros(num_agents)             
        score = 0
        agent.reset()
        for step in range(max_steps):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]     
            next_states = env_info.vector_observations   
            rewards = env_info.rewards                   
            dones = env_info.local_done                  
            agent.step(states, actions, rewards, next_states, dones, step)
            states = next_states
            scores_agents += rewards
            if np.any(dones):
                break
        score = np.mean(scores_agents)
        scores_hundred.append(score)
        average_score = np.mean(scores_hundred)
        scores.append(score)
        if score > highest_score:
            highest_score = score        
        print("Episode: ", i_episode)
        print("Min Score: {:.2f}  Max Score: {:.2f}".format(scores_agents.min(), scores_agents.max()))
        print("Score: {:.2f}".format(score))
        print("AvgScore: {:.2f}".format(average_score))
        print("Episode: {:2f} Score: {:.2f} Highest Score: {:.2f} Average Score: {:.2f}".format(i_episode, score, highest_score, average_score))
        if average_score > 30:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')            
            break
    return scores

scores = ddpg()