import math

class Agent:
    def __init__(self, env):
        self.env = env

    def play_episode(self, policy):
        state = self.env.reset()
        transitions = []
        while True:
            old_state = state
            action = policy.next_action(state)
            state, reward, is_done, _ = self.env.step(action)
            transitions.append(Transition(old_state, action, reward, state))
            if is_done:
                break
        return Episode(transitions)


class Transition:
    def __init__(self, old_state, action, reward, new_state):
        self.old_state = old_state
        self.action = action
        self.reward = reward
        self.new_state = new_state


class Episode:
    def __init__(self, transitions):
        self.transitions = transitions

    def total_reward(self):
        total_reward = 0
        for t in self.transitions:
            total_reward += t.reward
        return total_reward

    def to_string(self):
        out = '-'.join(str(t.new_state) for t in self.transitions)
        return "0-" + out + " # Reward:" + str(self.total_reward())

#    def calc_return_at(self, transition_index, gamma):
#        total_return = 0
#        for index, transition in enumerate(self.transitions[transition_index:]):
#            total_return += math.pow(gamma, index) * transition.reward
#        return total_return
