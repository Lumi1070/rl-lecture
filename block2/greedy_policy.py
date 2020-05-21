from block1.random_policy import RandomPolicy


class GreedyPolicy(RandomPolicy):

    def __init__(self, env):
        super().__init__(env)

    def greedy_update(self, env, state, q_values):
        best_action = q_values.index(max(q_values))

        action_probabilities = {}
        for i in range(0, env.env.nA):
            action_probabilities[i] = 1 if i == best_action else 0

        self.transitions[state] = action_probabilities

    def __str__(self):
        output = []
        for index, actions in enumerate(self.transitions):
            best_action = max(actions, key=lambda k: actions[k])
            output.append("state:" + str(index) + ", action:" + pprint_action(best_action))
        return "\n".join(output)


def pprint_action(action):
    if action == 0:
        return "\u2191"
    if action == 1:
        return "\u2192"
    if action == 2:
        return "\u2193"
    if action == 3:
        return "\u2190"
