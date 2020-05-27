**To install the environment, go to the gym-jobshop 
folder and run in a terminal:**\
`pip install -e .`

**Import the gym environment in Python with:**\
`import gym`\
`import gym_jobshop`\
`env = gym.make('jobshop-v0')`

**For version updates edit the following files:**\
`gym-jobshop/gym_jobshop/__init__.py `contains the ID of the environment\
`gym-jobshop/setup.py` contains the version number

Based on https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa