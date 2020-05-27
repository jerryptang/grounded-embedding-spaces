import argparse
import numpy as np

import config

def msqrt(m):
    '''Computes square root of positive semidefinite matrix m.'''
    e_val, e_vec = np.linalg.eigh(m)
    sqrts = np.sqrt(e_val.clip(0))
    ms = e_vec.dot(np.diag(sqrts)).dot(np.linalg.inv(e_vec))
    assert np.allclose(np.dot(ms, ms.T), m)
    return ms

def interpolate_grounded_embs(Av, V, A, sigma, lamb = 0.0):
    '''Interpolates grounded embeddings for nonvisual words.
    The columns of Av contain amodal embeddings of visual words.
    The columns of V contain visual embeddings of visual words.
    The columns of A contain amodal embeddings of all words.
    A sensory propagation model theta is fit to represent each amodal embedding in A by the amodal embeddings of visual words. theta is estimated using Tikhonov regression with prior covariance sigma and regularization coefficient lamb. The predicted grounded embeddings G, and the sensory propagation model theta, are returned. 
    '''
    C = msqrt(sigma)
    AvC = np.dot(Av, C)
    thetaT = np.linalg.inv(AvC.T.dot(AvC) + lamb * np.eye(Av.shape[1])).dot(AvC.T).dot(A)
    thetaT = np.dot(C, thetaT)
    G = np.dot(V, thetaT)
    return G, thetaT.T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('visual_embs_file', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()

    # load 3933 story words included in embedding space
    vocab = list(np.load(config.VOCAB_PATH))
    vocab_ids = dict((i, x) for x, i in enumerate(vocab))

    # load amodal embeddings for each vocab word
    print('loading amodal embeddings')
    amodal_embs = np.load(config.DISTRIBUTIONAL_EMBS_PATH)
    amodal_emb_dict = dict(zip(amodal_embs['sorted_vocab'], amodal_embs['embeddings'].T))
    A = np.array([amodal_emb_dict[x] for x in vocab]).T
    sigma_A = np.corrcoef(A.T)

    # load visual and amodal embeddings for each visual word
    print('loading visual embeddings')
    visual_embs = np.load(args.visual_embs_file)
    V = visual_embs['embeddings'].T
    Av = np.array([amodal_emb_dict[x] for x in visual_embs['visual_words']]).T

    # create spherical and visual priors for sensory propagation
    omega = {}
    omega['spherical'] = np.eye(len(visual_embs['visual_words']))
    omega['visual'] = np.corrcoef(V.T)

    # choose regularization coefficient to balance Ky Fan 1-norms of sigma_A and sigma_G
    print('setting regularization coefficient') 
    target_norm = max(np.linalg.eigh(sigma_A)[0])
    test_lambda = np.arange(100, 10000, 200)
    optimal_lambda = {}
    for om in omega:
        test_norm = {}
        for lamb in test_lambda: 
            test_G, _ = interpolate_grounded_embs(Av, V, A, omega[om], lamb = lamb)
            test_sigma_G = np.corrcoef(test_G.T)
            test_norm[lamb] = max(np.linalg.eigh(test_sigma_G)[0])
            if test_norm[lamb] > target_norm: break
        optimal_lambda[om] = sorted([(k, abs(v - target_norm)) for k, v in test_norm.items()], key = lambda x : x[1])[0][0]

    # interpolate visually grounded embeddings using optimal regularization coefficients
    print('creating visually grounded embeddings') 
    sigma_G, theta = {}, {}
    for om in omega:
        G, theta[om] = interpolate_grounded_embs(Av, V, A, omega[om], lamb = optimal_lambda[om])
        sigma_G[om] = np.corrcoef(G.T)

    np.savez(args.outfile,
        vocab=vocab,
        visual_words=visual_embs['visual_words'],
        sigma_A=sigma_A,
        sigma_G=(sigma_G['visual'] + sigma_G['spherical']) / 2,
        theta=(theta['visual'] + theta['spherical']) / 2)