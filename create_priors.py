import argparse
import numpy as np

import config

def msqrt(m):
    '''Computes square root of positive semidefinite matrix m.'''
    e_val, e_vec = np.linalg.eigh(m)
    sqrts = np.sqrt(e_val.clip(0))
    ms = e_vec.dot(np.diag(sqrts)).dot(np.linalg.inv(e_vec))
    assert np.allclose(ms.dot(ms.T), m)
    return ms

def interpolate_grounded_embs(Lv, C, L, sigma, lamb = 0.0):
    '''Interpolates visual embeddings for visual and nonvisual words.
    The columns of Lv contain linguistic embeddings of visual words.
    The columns of C contain CNN embeddings of visual words.
    The columns of L contain linguistic embeddings of all words.
    A sensory propagation model theta is fit to represent each linguistic embedding in L by the linguistic embeddings of visual words. theta is estimated using Tikhonov regression with prior covariance sigma and regularization coefficient lamb. The predicted visual embeddings V, and the sensory propagation model theta, are returned. 
    '''
    MS = msqrt(sigma)
    LvMS = Lv.dot(MS)
    thetaT = np.linalg.inv(LvMS.T.dot(LvMS) + lamb * np.eye(Lv.shape[1])).dot(LvMS.T).dot(L)
    thetaT = MS.dot(thetaT)
    V = C.dot(thetaT)
    return V, thetaT.T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cnn_embs_file', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()

    # load 3933 stimulus words included in embedding space
    vocab = list(np.load(config.VOCAB_PATH))

    # load linguistic embeddings for each vocab word
    print('loading linguistic embeddings')
    linguistic_embs = np.load(config.DISTRIBUTIONAL_EMBS_PATH)
    linguistic_emb_dict = dict(zip(linguistic_embs['sorted_vocab'], linguistic_embs['embeddings'].T))
    L = np.array([linguistic_emb_dict[x] for x in vocab]).T
    sigma_L = np.corrcoef(L.T)

    # load CNN and linguistic embeddings for each visual word
    print('loading CNN embeddings')
    cnn_embs = np.load(args.cnn_embs_file)
    C = cnn_embs['embeddings'].T
    Lv = np.array([linguistic_emb_dict[x] for x in cnn_embs['visual_words']]).T

    # create spherical and CNN priors for sensory propagation
    omega = {}
    omega['spherical'] = np.eye(len(cnn_embs['visual_words']))
    omega['cnn'] = np.corrcoef(C.T)

    # choose regularization coefficient to balance Ky Fan 1-norms of sigma_L and sigma_G
    print('setting regularization coefficient') 
    target_norm = max(np.linalg.eigh(sigma_L)[0])
    test_lambda = np.arange(100, 10000, 200)
    optimal_lambda = {}
    for om in omega:
        test_norm = {}
        for lamb in test_lambda: 
            test_V, _ = interpolate_grounded_embs(Lv, C, L, omega[om], lamb = lamb)
            test_sigma_V = np.corrcoef(test_V.T)
            test_norm[lamb] = max(np.linalg.eigh(test_sigma_V)[0])
            if test_norm[lamb] > target_norm: break
        optimal_lambda[om] = sorted([(k, abs(v - target_norm)) for k, v in test_norm.items()], key = lambda x : x[1])[0][0]

    # interpolate visual embeddings using optimal regularization coefficients
    print('creating visual embeddings') 
    sigma_V, theta = {}, {}
    for om in omega:
        V, theta[om] = interpolate_grounded_embs(Lv, C, L, omega[om], lamb = optimal_lambda[om])
        sigma_V[om] = np.corrcoef(V.T)

    np.savez(args.outfile,
        vocab=vocab,
        visual_words=cnn_embs['visual_words'],
        sigma_L=sigma_L,
        sigma_V=(sigma_V['cnn'] + sigma_V['spherical']) / 2,
        theta=(theta['cnn'] + theta['spherical']) / 2)