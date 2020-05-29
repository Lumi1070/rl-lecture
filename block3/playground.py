from block1.agent import Agent
from block1.random_policy import RandomPolicy
from block2.policy_iteration import policy_iteration
from block3.monte_carlo import policy_iteration_mc
from block3.td_zero import policy_iteration_td_zero

import gym

ENV_NAME = "FrozenLake-v0"
GAMMA = 1
env = gym.make(ENV_NAME)
agent = Agent(env)

no_episodes = 1000

print("\n")
print(f"## Random Policy - {no_episodes} episodes ##")
random_policy = RandomPolicy(env)
total_reward = 0.0
for i in range(0, no_episodes):
    episode = agent.play_episode(random_policy)
    total_reward += episode.total_reward()
print("Total reward: " + str(total_reward))

print("\n")
print("## Policy Iteration ##")
best_policy = policy_iteration(env)
total_reward = 0.0
for i in range(0, 1000):
    episode = agent.play_episode(best_policy)
    total_reward += episode.total_reward()
print("Total reward: " + str(total_reward))


print("\n")
print("## Policy Iteration with mc-evaluation ##")
best_policy = policy_iteration_mc(env, agent)
total_reward = 0.0
for i in range(0, 1000):
    episode = agent.play_episode(best_policy)
    total_reward += episode.total_reward()
print("Total reward: " + str(total_reward))

print("\n")
print("## Policy Iteration with td-zero ##")
best_policy = policy_iteration_td_zero(env, agent)
total_reward = 0.0
for i in range(0, 1000):
    episode = agent.play_episode(best_policy)
    total_reward += episode.total_reward()
print("Total reward: " + str(total_reward))
