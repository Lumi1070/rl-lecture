GAMMA = 1
ALPHA = 0.2


def every_visit_mc(env, agent, policy, samples):

    state_values = [0] * env.observation_space.n
    for episodes in range(0, samples):

        episode = agent.play_episode(policy)
        state_rewards = [0] * env.observation_space.n
        state_visits = [0] * env.observation_space.n
        for i, transition in enumerate(episode.transitions):
            state_rewards[transition.old_state] += episode.calc_return_at(i, GAMMA)
            state_visits[transition.old_state] += 1

        # update state values
        for i, value in enumerate(state_values):
            if state_visits[i] > 0:
                state_return = state_rewards[i] / state_visits[i]
                state_values[i] = state_values[i] + ALPHA*(state_return - state_values[i])

    return state_values

