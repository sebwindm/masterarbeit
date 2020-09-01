**To release or not to release - project overview**\
There are two components in this project:
1. A custom environment for http://gym.openai.com/ containing the simulation of a job shop production system. 
   This environment allows a reinforcement learning (RL) agent to interact with the production system through a structured interface.
   The RL agent can control the production capacity through this interface and receives back a reward (in form of incurred costs).
   On the long run, the agent can learn what its actions caused inside the environment due to the feedback it received for each action.
   This allows the agent to make better decisions in the future, with the goal being that the agent can beat established heuristics or human decision making. 


2. An implementation of a reinforcement learning agent for the environmentfrom above, based on the Deep Q-Learning algorithm
   of https://stable-baselines3.readthedocs.io/en/master/.
   The original algorithm has received additional functionality by adding Average Reward Adjusted Discounted Reinforcement Learning,
   a technique that allows RL agents to handle rewards of less established shapes, i.e. periodically incurred costs. 


** Installation:

See gym-jobshop/readme_gym_jobshop.md for installation
instructions for the custom environment.

** Installation on ArchLinux

You have to use python version 3.6 as tensorflow 1.5 is a dependency

    yay -S python36   # install python 3.6
    virtualenv --python=`which python3.6` env   # create virtual environment
    source env/bin/activate
    pip install -r requirements.txt
