**Project overview**\
There are two components:
1. A custom environment for http://gym.openai.com/ containing the simulation of a job shop production system. 
   This environment allows a reinforcement learning (RL) agent to interact with the production system through a structured interface.
   The RL agent can control the production capacity through this interface and receives back a reward (in form of incurred costs).
   On the long run, the agent can learn what its actions caused inside the environment due to the feedback it received for each action.
   This allows the agent to make better decisions in the future, with the goal being that the agent can beat established heuristics or human decision making. 


2. An implementation of a reinforcement learning agent for the environmentfrom above, based on the Deep Q-Learning algorithm
   of https://stable-baselines3.readthedocs.io/en/master/.
   The original algorithm has received additional functionality by adding Average Reward Adjusted Discounted Reinforcement Learning,
   a technique that allows RL agents to handle rewards of less established shapes, i.e. periodically incurred costs. 


**Installation**\
Requirements:
* Python 3.6+
* PyTorch 1.4+  (**not** the CUDA version)
* Stable Baselines 3 (**not** 2)
* Preferably a Linux-based system (otherwise you can't use the
visualisation plots and some stuff might not work)


To install the gym jobshop environment, go to the gym-jobshop 
folder and run in a terminal:\
`pip install -e .`

Import the gym environment to your Python script:\
`import gym`\
`import gym_jobshop`

Setup the environment:\
`env = gym.make('jobshop-v0')`

For documentation on how to work with Gym environments,
go to https://gym.openai.com/docs/

**Troubleshooting**

Known issues:
* Stable Baselines 3 must be on version 0.8.0.
As of 09/2020 the current SB3 version on pip is 0.8.0, which is the correct one.
Version 0.9 (bleeding edge installation) will cause a `TypeError` when training.
* PyTorch should be installed from https://pytorch.org/get-started/locally/
e.g. with the following command:\
`pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
`. Any CUDA version will not work.




Installation on ArchLinux (**outdated**)

You have to use python version 3.6 as tensorflow 1.5 is a dependency

    yay -S python36   # install python 3.6
    virtualenv --python=`which python3.6` env   # create virtual environment
    source env/bin/activate
    pip install -r requirements.txt`
