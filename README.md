# Simplified Banana Collection

### Introduction

This project contains an agent that collects yellow bananas in a simplified, single-agent version of Unity’s Banana Collector environment.

The environment is defined by a continuous state space with dimension size of 37 corresponding to the velocity of the agent, ray-based perception of objects in the agent’s forward direction, and other measures.

The actions the agent may take are limited to 4 discrete options: go forward (0), go backwards (1), turn left (2) or turn right (3).

The goal of the agent is to pick up as many yellow bananas as possible while avoiding blue ones.  To achieve this, a reward of +1 is given for picking up a yellow banana, while a reward of -1 is given for picking up a blue one.

The environment is considered solved when the agent garners an average score of at least 13 over a series of 100 episodes.

The code is a slightly modified from of code provided by Udacity in their Deep Reinforcement Learning Nanodegree Program (Deep Q-Learning Algorithm exercise, Lesson 2, Deep Q Networks.)

### Getting Started
1. [Download](https://www.python.org/downloads/) and install Python 3.6 or higher if not already installed.
2. Install conda if not already installed.  To install conda, follow these [instructions](https://conda.io/docs/user-guide/install/index.html)
3. Create and activate a new conda environment
```
conda create -n p1_banana python=3.6
conda activate p1_banana
```
3. Clone this GitHub repository
```
git clone https://github.com/lynnolson/p1_banana.git
```
4. Change to the p1_banana directory and install python dependencies by running setup.py
```
cd p1_banana
python setup.py install
```
If you are going to run this on a mac, you may also need to add the file .matplotlib/matplotlibrc with the following contents under your home directory if it does not already exist.
```
backend: TkAgg
```
5. Download the Banana collection environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

6. Place the environment zip in the `p1_banana/` folder, and unzip (or decompress) the file.

### Training Instructions

To train the agent, run train.py

```
python train.py -banana_env_path Banana.app -dqn_chck_pt_path dqn_model_weights.pt
```

To save a plot of the scores over time (successive episodes), set the argument plot_path to a specific file

```
python train.py -banana_env_path Banana.app -dqn_chck_pt_path dqn_model_weights.pt -plot_path score_per_episode.png
```
The model weights are saved in the file specified by dqn_chck_pt_path.  Currently there is no mechanism to recreate the model from these parameters.
When you are done, deactivate the conda environment:
```
conda deactivate
```
### Note
The whole procedure above has only been tested on Mac OS X El Capitan.
