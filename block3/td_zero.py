from block2.greedy_policy import GreedyPolicy
from block2.policy_iteration import eval_action

GAMMA = 1
ALPHA = 0.5


def td_zero(env, agent, policy):

    episodes = []
    for i in range(0, 1000):
        episodes.append(agent.play_episode(policy))

    # todo: update directly after each time step
    state_values = [0] * env.observation_space.n
    for episode in episodes:
        for transition in episode.transitions:
            old_state = transition.old_state
            reward = transition.reward
            new_state = transition.new_state

            td_target = reward + GAMMA * state_values[new_state]
            state_values[old_state] += ALPHA * (td_target - state_values[old_state])

    return state_values


def policy_iteration_td_zero(env, agent):
    iteration = 0
    policy = GreedyPolicy(env) # initially this is a random policy
    while True:
        iteration += 1
        old_policy = policy
        state_values = td_zero(env, agent, policy)
        policy = GreedyPolicy(env)
        for state in range(0, env.observation_space.n):

            # todo: learn Q-values with SARSA instead of using eval_action function
            q_values = [eval_action(env, state, action, state_values) for action in range(0, env.action_space.n)]
            policy.greedy_update(env, state, q_values)

        if policy == old_policy:
            break

    print("Converged after " + str(iteration) + " iterations.")
    return policy
