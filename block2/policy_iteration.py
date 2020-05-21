from block2.greedy_policy import GreedyPolicy

GAMMA = 1


def eval_action(env, state, action, state_values):
    targets = env.env.P[state][action]
    action_value = 0
    for prob, target_state, reward, _ in targets:
        action_value += prob * (reward + GAMMA * state_values[target_state])
    return action_value


def update_state_values(env, state_values, policy):
    new_state_values = [0] * env.observation_space.n
    for state in range(0, env.env.nS):
        for action in range(0, env.action_space.n):
            action_value = eval_action(env, state, action, state_values)
            new_state_values[state] += policy.transition_proba(state, action) * action_value
    return new_state_values


def eval_state_values(env, policy):
    state_values = [0] * env.observation_space.n
    epsilon = 0.001
    delta = 1
    iterations = 0
    while delta > epsilon:
        new_state_values = update_state_values(env, state_values, policy)
        delta = 0
        for i in range(0, env.observation_space.n):
            diff = abs(new_state_values[i] - state_values[i])
            delta = max(delta, diff)
        state_values = new_state_values
        iterations += 1

    # print("State value evaluation converged after " + str(iterations) + " iterations with: " +str(state_values))
    return state_values


def policy_iteration(env):
    iteration = 0
    policy = GreedyPolicy(env) # initially this is a random policy
    while True:
        iteration += 1
        old_policy = policy
        state_values = eval_state_values(env, policy)
        policy = GreedyPolicy(env)
        for state in range(0, env.observation_space.n):
            q_values = [eval_action(env, state, action, state_values) for action in range(0, env.action_space.n)]
            policy.greedy_update(env, state, q_values)

        if policy == old_policy:
            break

    print("Converged after " + str(iteration) + " iterations.")
    return policy
