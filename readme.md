**To release or not to release - project overview**\
There are two components in this project:
1. Custom environment for Gym (http://gym.openai.com/) containing the simulation
of a production system (system parameters are based on literature).
This environment allows a reinforcement learning agent to interact with the production
system and learn from its interactions with the system.
See gym-jobshop/readme_gym_jobshop.md for installation
instructions for the custom environment.

2. An example implementation of a reinforcement learning agent for this environment


** Installation on ArchLinux

You have to use python version 3.6 as tensorflow 1.5 is a dependency

    yay -S python36   # install python 3.6
    virtualenv --python=`which python3.6` env   # create virtual environment
    source env/bin/activate
    pip install -r requirements.txt
