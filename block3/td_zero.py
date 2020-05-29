GAMMA = 1
ALPHA = 0.5


def td_zero(env, agent, policy, samples):

    state_values = [0] * env.observation_space.n

    for i in range(0, samples):
        episode = agent.play_episode(policy)
        for t in episode.transitions:
            td_target = t.reward + GAMMA * state_values[t.new_state]
            state_values[t.old_state] += ALPHA * (td_target - state_values[t.old_state])

    return state_values
