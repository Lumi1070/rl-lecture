from block1.policy import Policy


class RandomPolicy(Policy):

    def __init__(self, env):

        action_probabilities = {}
        for i in range(0, env.env.nA):
            action_probabilities[i] = 1 / env.env.nA

        transitions = []
        for i in range(0, env.env.nS):
            transitions.append(action_probabilities.copy())

        super().__init__(transitions)
