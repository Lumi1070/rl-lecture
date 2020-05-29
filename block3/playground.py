from block1.agent import Agent
from block1.random_policy import RandomPolicy
from block2.policy_iteration import eval_state_values
from block3.monte_carlo import every_visit_mc
from block3.td_zero import td_zero

import gym

ENV_NAME = "FrozenLake-v0"
GAMMA = 1
env = gym.make(ENV_NAME)
agent = Agent(env)
print("## Frozen Lake ##")
env.render()

random_policy = RandomPolicy(env)

print("\n")
print("## Policy Evaluation on MDP##")
state_values = eval_state_values(env, random_policy)
print(f"State values:")
print([f'{x:.3f}' for x in state_values[0:4]])
print([f'{x:.3f}' for x in state_values[4:8]])
print([f'{x:.3f}' for x in state_values[8:12]])
print([f'{x:.3f}' for x in state_values[12:16]])

print("\n")
print("## Policy Evaluation with mc-evaluation ##")
state_values = every_visit_mc(env, agent, random_policy, 1000)
print(f"State values:")
print([f'{x:.3f}' for x in state_values[0:4]])
print([f'{x:.3f}' for x in state_values[4:8]])
print([f'{x:.3f}' for x in state_values[8:12]])
print([f'{x:.3f}' for x in state_values[12:16]])


print("\n")
print("## Policy Evaluation with temporal differencing ##")
state_values = every_visit_mc(env, agent, random_policy, 1000)
print(f"State values:")
print([f'{x:.3f}' for x in state_values[0:4]])
print([f'{x:.3f}' for x in state_values[4:8]])
print([f'{x:.3f}' for x in state_values[8:12]])
print([f'{x:.3f}' for x in state_values[12:16]])
