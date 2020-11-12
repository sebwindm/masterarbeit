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

**If you're new to this environment:**
* The most important files to look at are `global_settings.py`, `jobshop_env.py `
and `agents_training_and_evaluation.py`
* Change the planned release date slack for the BIL order release in `global_settings.py`
with `planned_release_date_multiplier`
* Change the overtime setup at two places inside `global_settings.py`. 
First change `overtime_multiplier_2` and `overtime_multiplier_3` to adjust 
the overtime's influence on processing time. Then change `cost_for_action_1` and `cost_for_action_2`
to adjust the cost of overtime. 
All these values need to be changed for switching between the experimental designs.
* Change the order release policy with `order_release_policy`. BIL and periodic release are supported.
* Only job shop but not flow shop is supported. There is however a lot of work already done for a flow shop to work,
since this environment was a flow shop at first. You will not need to change a lot to change it to a flow shop.
* Demand and processing times use exponential distributions. If you want to switch to 
uniform distributions, change `demand_distribution` and `processing_time_distribution` accordingly
* Change the due date slack with `due_date_multiplier`
* Change exponential demand wih `next_order_arrival_exponential_rate_parameter`
* Change uniform demand with `next_order_arrival_lower_bound` + `next_order_arrival_upper_bound`
* Machine processing times are set in `main.py` -> `setup_environment()`
* Both the RL agents and the job shop environment have a fixed random number stream
* Change cost settings at `cost_per_item_in_shopfloor`, `cost_per_item_in_fgi`,
`cost_per_late_item`, `overtime_base_cost`
