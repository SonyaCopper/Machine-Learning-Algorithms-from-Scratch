import numpy as np



class Viterbi():

    def __init__(self, initial_probability, transitions, emission):
        self.n = initial_probability.shape[0]
        self.initial_probability = initial_probability
        self.transitions = transitions
        self.emission = emission
        assert self.initial_probability.shape == (self.n,)
        assert self.transitions.shape == (self.n, self.n)
        assert self.emission.shape[0] == self.n

    def run(self, observations):
        states = np.zeros((self.transitions.shape[0], len(observations)))
        probabilities = np.zeros((self.transitions.shape[0], len(observations)))
        probabilities[:, 0] = self.initial_probability * self.emission[:, observations[0]]

        for t in range(1, len(observations)):
            for s in range(states.shape[0]):
                trellis = probabilities[:, t - 1] * self.transitions[:, s] * self.emission[s, observations[t]]
                states[s, t - 1] = np.argmax(trellis)
                probabilities[s, t] = np.max(trellis)
        highest_probability = np.argmax(probabilities[:, -1])
        most_probable_path = [highest_probability]

        for t in range((len(observations) - 1), 0, -1):
            mt = int(states[highest_probability, t - 1])
            most_probable_path.append(mt)
            highest_probability = mt

        return most_probable_path[::-1]


pi = np.array([0.04, 0.02, 0.06, 0.04, 0.11, 0.11, 0.01, 0.09, 0.03, 0.05, 0.06, 0.11, 0.05, 0.11, 0.03, 0.08])

trans = np.array([[0.08, 0.02, 0.10, 0.05, 0.07, 0.08, 0.07, 0.04, 0.08, 0.10, 0.07, 0.02, 0.01, 0.10, 0.09, 0.01],
                  [0.06, 0.10, 0.11, 0.01, 0.04, 0.11, 0.04, 0.07, 0.08, 0.10, 0.08, 0.02, 0.09, 0.05, 0.02, 0.02],
                  [0.08, 0.07, 0.08, 0.07, 0.01, 0.03, 0.10, 0.02, 0.07, 0.03, 0.06, 0.08, 0.03, 0.10, 0.10, 0.08],
                  [0.08, 0.04, 0.04, 0.05, 0.07, 0.08, 0.01, 0.08, 0.10, 0.07, 0.11, 0.01, 0.05, 0.04, 0.11, 0.06],
                  [0.03, 0.03, 0.08, 0.10, 0.11, 0.04, 0.06, 0.03, 0.03, 0.08, 0.03, 0.07, 0.10, 0.11, 0.07, 0.03],
                  [0.02, 0.05, 0.01, 0.09, 0.05, 0.09, 0.05, 0.12, 0.09, 0.07, 0.01, 0.07, 0.05, 0.05, 0.11, 0.06],
                  [0.11, 0.05, 0.10, 0.07, 0.01, 0.08, 0.05, 0.03, 0.03, 0.10, 0.01, 0.10, 0.08, 0.09, 0.07, 0.02],
                  [0.03, 0.02, 0.16, 0.01, 0.05, 0.01, 0.14, 0.14, 0.02, 0.05, 0.01, 0.09, 0.07, 0.14, 0.03, 0.01],
                  [0.01, 0.09, 0.13, 0.01, 0.02, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.06, 0.11, 0.06, 0.03, 0.14],
                  [0.09, 0.03, 0.04, 0.05, 0.04, 0.03, 0.12, 0.04, 0.07, 0.02, 0.07, 0.10, 0.11, 0.03, 0.06, 0.09],
                  [0.09, 0.04, 0.06, 0.06, 0.05, 0.07, 0.05, 0.01, 0.05, 0.10, 0.04, 0.08, 0.05, 0.08, 0.08, 0.10],
                  [0.07, 0.06, 0.01, 0.07, 0.06, 0.09, 0.01, 0.06, 0.07, 0.07, 0.08, 0.06, 0.01, 0.11, 0.09, 0.05],
                  [0.03, 0.04, 0.06, 0.06, 0.06, 0.05, 0.02, 0.10, 0.11, 0.07, 0.09, 0.05, 0.05, 0.05, 0.11, 0.08],
                  [0.04, 0.03, 0.04, 0.09, 0.10, 0.09, 0.08, 0.06, 0.04, 0.07, 0.09, 0.02, 0.05, 0.08, 0.04, 0.09],
                  [0.05, 0.07, 0.02, 0.08, 0.06, 0.08, 0.05, 0.05, 0.07, 0.06, 0.10, 0.07, 0.03, 0.05, 0.06, 0.10],
                  [0.11, 0.03, 0.02, 0.11, 0.11, 0.01, 0.02, 0.08, 0.05, 0.08, 0.11, 0.03, 0.02, 0.10, 0.01, 0.11]])
obs = np.array(
    [[0.01, 0.99], [0.58, 0.42], [0.48, 0.52], [0.58, 0.42], [0.37, 0.63], [0.33, 0.67], [0.51, 0.49], [0.28, 0.72],
     [0.35, 0.65], [0.61, 0.39], [0.97, 0.03], [0.87, 0.13], [0.46, 0.54], [0.55, 0.45], [0.23, 0.77], [0.76, 0.24]])
data = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]])
v = Viterbi(pi, trans, obs)
mst_path1 = v.run(data[1])
mst_path2 = v.run(data[0])
print(mst_path1)
print(mst_path2)
