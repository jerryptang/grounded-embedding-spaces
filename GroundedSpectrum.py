import numpy as np

class GroundedSpectrum(object):
    
    def __init__(self, prior_path, conc_scores_path): 
        data = np.load(prior_path)
        self.sigma_A, self.sigma_G, self.theta = data['sigma_A'], data['sigma_G'], data['theta']
        self.vocab, self.visual_words = data['vocab'], data['visual_words']
        self.vocab_ids = dict((i, x) for x, i in enumerate(self.vocab))
        self.conc_scores = np.load(conc_scores_path).clip(0.01, 0.99)

    def grounding_words(self, w):
        """Identify the most visual words that are most strongly aassociated with w under the sensory propagation model."""
        theta_w = self.theta[self.vocab_ids[w]]
        return sorted(zip(self.visual_words, theta_w), key = lambda x : -x[1])

    def sigma(self, b):
        """Interpolate between sigma_A and sigma_G.
        The grounded and amodal embeddings of each word are weighted by a function alpha of the word's concreteness. The bias parameter b controls the total amount of visual information in the joint space."""
        alphas = np.array([self._sigmoid(self._inv_sigmoid(x) + b) for x in self.conc_scores])
        grounded_wts = np.diag(np.sqrt(alphas))
        amodal_wts = np.diag(np.sqrt(1 - alphas))
        return amodal_wts.dot(self.sigma_A).dot(amodal_wts) + grounded_wts.dot(self.sigma_G).dot(grounded_wts)
    
    def similarity(self, w1, w2, b = 0):
        id1, id2 = self.vocab_ids[w1], self.vocab_ids[w2]
        c1 = self._sigmoid(self._inv_sigmoid(self.conc_scores[id1]) + b)
        c2 = self._sigmoid(self._inv_sigmoid(self.conc_scores[id2]) + b)
        return np.sqrt(c1 * c2) * self.sigma_G[id1, id2] + np.sqrt((1 - c1) * (1 - c2)) * self.sigma_A[id1, id2]

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _inv_sigmoid(x):
        if x >= 1.0: return np.inf
        if x == 0.0: return -np.inf
        return np.log(x/(1-x))