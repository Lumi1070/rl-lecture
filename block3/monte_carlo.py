from block2.greedy_policy import GreedyPolicy
from block2.policy_iteration import eval_action

GAMMA = 1


def first_visit_mc(env, agent, policy):
    episodes = []
    for i in range(0, 10000):
        episodes.append(agent.play_episode(policy))

    # todo: update directly after each episode using running mean
    state_rewards = [0] * env.observation_space.n
    state_visits = [0] * env.observation_space.n
    for episode in episodes:
        visited = set()
        for i, transition in enumerate(episode.transitions):
            if transition.old_state not in visited:
                state_rewards[transition.old_state] += episode.calc_return_at(i, GAMMA)  # only valid for gamma=1
                state_visits[transition.old_state] += 1
                visited.add(transition.old_state)

    state_values = []
    for i in range(0, env.observation_space.n):
        state_value = state_rewards[i]/state_visits[i] if state_visits[i] != 0 else 0
        state_values.append(state_value)
    return state_values


def policy_iteration_mc(env, agent):
    iteration = 0
    policy = GreedyPolicy(env) # initially this is a random policy
    while True:
        iteration += 1
        old_policy = policy
        state_values = first_visit_mc(env, agent, policy)
        policy = GreedyPolicy(env)
        for state in range(0, env.observation_space.n):
            # todo: learn Q-values instead of using eval_action function
            q_values = [eval_action(env, state, action, state_values) for action in range(0, env.action_space.n)]
            policy.greedy_update(env, state, q_values)

        if policy == old_policy:
            break

    print("Converged after " + str(iteration) + " iterations.")
    return policy
