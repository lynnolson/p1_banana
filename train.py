"""
    Train an agent to collect yellow bananas and avoid blue ones in a Unity
    environment.  The goal is to get a score greater than 13 in 100
    consecutive episodes.
"""
import argparse
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch

from unityagents import UnityEnvironment

from dqn_agent import Agent

def dqn(agent,
        dqn_chck_pt_path,
        n_episodes=1800,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995):
    """Train the model using Deep Q-Learning (DQN).
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        
    Returns
    =======
        Average score per 100 episodes
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for _ in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), dqn_chck_pt_path)
            break
    return scores

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-banana_env_path", required=True, help="Path Unity banana environment app")
    parser.add_argument("-dqn_chck_pt_path", required=True, help="File to save DQN model parameters")
    parser.add_argument("-plot_path", default=None, help="File to save plot of score per episode")
    
    # Training parameters
    parser.add_argument('-n_episodes', type=int, default=1800, help='maximum number of training episodes')
    parser.add_argument('-max_t', type=int, default=1000, help='maximum number of timesteps per episode')
    parser.add_argument('-eps_start', type=float, default=1.0, help='starting value of epsilon')
    parser.add_argument('-eps_end', type=float, default=0.01, help='minimum value of epsilon')
    parser.add_argument('-eps_decay', type=float, default=0.995, help='multiplicative factor for decreasing epsilon')
    parser.add_argument('-batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('-lr', type=float, default=5e-4 , help='learning rate')
    
    # Agent-related parameters.
    parser.add_argument('-gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('-buf_size', type=int, default=100000, help='replay buffer size')
    parser.add_argument('-tau', type=float, default=1e-3, help='for soft update of target parameters')
    parser.add_argument('-update_t', type=int, default=4, help='how often to update the network')
    
    # Q Network (model) parameters.
    parser.add_argument('-fc1_units', type=int, default=64, help='Number of nodes in first hidden layer')
    parser.add_argument('-fc2_units', type=int, default=64, help='Number of nodes in second hidden layer')
    
    parser.add_argument('-seed', type=int, default=777, help="Random seed")
    opt = parser.parse_args()

    # Load the environment for collecting bananas.
    env = UnityEnvironment(file_name='banana_env_path')
    
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # Determine the size of the action and state spaces.
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size 
    state = env_info.vector_observations[0]
    state_size = len(state)

    # Create our agent.
    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  buf_size=opt.buf_size,
                  gamma=opt.gamma,
                  tau=opt.tau,
                  update_t=opt.update_t,
                  lr=opt.lr,
                  batch_size=opt.batch_size,
                  fc1_units=opt.fc1_units,
                  fc2_units=opt.fc2_units,
                  seed=opt.seed)
    
    # Train the Deep Q Network (DQN)
    scores = dqn(agent,
                 opt.dqn_chck_pt_path,
                 opt.n_episodes,
                 opt.max_t,
                 opt.eps_start,
                 opt.eps_end,
                 opt.eps_decay)
    
    # Optionally create and save a plot of score versus episode number.
    if opt.plot_path is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(opt.plot_path)
    
    env.close()