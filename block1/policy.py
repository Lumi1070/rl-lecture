import random


class Policy:
    def __init__(self, transitions):
        self.transitions = transitions

    def transition_proba(self, state, action):
        return self.transitions[state][action]

    def next_action(self, state):
        actions = self.transitions[state].keys()
        probabilities = self.transitions[state].values()
        return random.choices(list(actions), list(probabilities))[0]

    def __eq__(self, other):
        return self.transitions == other.transitions

    def __str__(self):
        return str(self.transitions)
