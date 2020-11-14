import numpy as np

class SemanticSpace(object):
    
    def __init__(self, prior_path, conc_scores_path): 
        data = np.load(prior_path)
        self.sigma_L, self.sigma_V, self.theta = data['sigma_L'], data['sigma_V'], data['theta']
        self.vocab, self.visual_words = data['vocab'], data['visual_words']
        self.word2id = dict((i, x) for x, i in enumerate(self.vocab))
        self.conc_scores = np.load(conc_scores_path).clip(0.01, 0.99)

    def sigma(self, b):
        """Create semantic embedding space by combining visual and linguistic embedding spaces.
        The visual and linguistic embeddings of each word are weighted by a function alpha of the word's concreteness. The bias parameter b controls the total amount of visual information in the semantic embedding space."""
        alphas = np.array([self._sigmoid(self._inv_sigmoid(c) + b) for c in self.conc_scores])
        visual_wts = np.diag(np.sqrt(alphas))
        linguistic_wts = np.diag(np.sqrt(1 - alphas))
        return linguistic_wts.dot(self.sigma_L).dot(linguistic_wts) + visual_wts.dot(self.sigma_V).dot(visual_wts)
    
    def similarity(self, w1, w2, b = 0):
        id1, id2 = self.word2id[w1], self.word2id[w2]
        alpha1 = self._sigmoid(self._inv_sigmoid(self.conc_scores[id1]) + b)
        alpha2 = self._sigmoid(self._inv_sigmoid(self.conc_scores[id2]) + b)
        return np.sqrt(alpha1 * alpha2) * self.sigma_V[id1, id2] + np.sqrt((1 - alpha1) * (1 - alpha2)) * self.sigma_L[id1, id2]

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _inv_sigmoid(x):
        if x >= 1.0: return np.inf
        if x <= 0.0: return -np.inf
        return np.log(x/(1-x))