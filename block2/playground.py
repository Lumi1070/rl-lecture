from block1.agent import Agent
from block1.random_policy import RandomPolicy
from block2.policy_iteration import policy_iteration
import gym

ENV_NAME = "FrozenLake-v0"
GAMMA = 1
env = gym.make(ENV_NAME)
agent = Agent(env)

print("\n")
print("## Random Policy ##")
random_policy = RandomPolicy(env)
total_reward = 0.0
for i in range(0, 1000):
    episode = agent.play_episode(random_policy)
    total_reward += episode.total_reward()
print("Total reward: " + str(total_reward))

print("\n")
print("## Policy Iteration ##")
best_policy = policy_iteration(env)
total_reward = 0.0
iteration_max = 0
for i in range(0, 1000):
    episode = agent.play_episode(best_policy)
    total_reward += episode.total_reward()
print("Total reward: " + str(total_reward))
