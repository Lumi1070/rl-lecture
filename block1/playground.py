from block1.agent import Agent
from block1.random_policy import RandomPolicy
import gym

ENV_NAME = "FrozenLake-v0"
GAMMA = 1
env = gym.make(ENV_NAME)
agent = Agent(env)
print("## Frozen Lake ##")
env.render()

print("\n ## Random Policy ##")
random_policy = RandomPolicy(env)
total_reward = 0.0
for i in range(0, 10):
    episode = agent.play_episode(random_policy)
    total_reward += episode.total_reward()
    print(f"Episode-{i}: {episode}")
print(f"Total reward: {total_reward}")
