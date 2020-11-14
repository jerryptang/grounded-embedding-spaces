import argparse
import numpy as np

import config

def similarity_scores(w, sigma, vocab, word2id):
    '''Returns the similarity of each word in vocab to w, sorted in descending order.'''
    vec = sigma[word2id[w]]
    return sorted(zip(vocab, vec), key = lambda x : -x[1])

def interpolate_conc_scores(vocab, word2id, sigma_L, conc_ratings, top_nn = 15):
    '''Compute concreteness scores for each word in vocab, using linguistic similarities to interpolate scores for missing words.'''
    scores = {}
    for w in vocab:
        # concreteness score of w
        if w in conc_ratings: w_score = conc_ratings[w]**2
        else: w_score = 0
        # mean concreteness score of neighbors
        nn_correlations = similarity_scores(w, sigma_L, vocab, word2id)
        nn_score = np.mean([conc_ratings[nn[0]] for nn in nn_correlations if nn[0] in conc_ratings][:top_nn])**2
        scores[w] = max(w_score, nn_score)
    return np.array([scores[w] for w in vocab])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('priors_file', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()

    # load story words and linguistic embedding space
    data = np.load(args.priors_file)
    vocab = data['vocab']
    word2id = dict((i, x) for x, i in enumerate(vocab))
    sigma_L = data['sigma_L']

    # interpolate concreteness scores from concreteness ratings
    conc_ratings = np.load(config.CONC_RATINGS_PATH)
    conc_ratings_dict = dict(zip(conc_ratings['vocab'], conc_ratings['ratings']))
    conc_scores = interpolate_conc_scores(vocab, word2id, sigma_L, conc_ratings_dict, top_nn = 15)

    np.save(args.outfile, conc_scores)