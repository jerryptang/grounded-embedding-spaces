# Grounded Embedding Spaces

This repository contains code for creating visually grounded word embedding spaces used in the paper "Cortical Representations of Concrete and Abstract Concepts are Grounded in Visual Features" by Jerry Tang, Amanda I. Lebel, and Alexander G. Huth.  

## Usage

1. Extract visual embeddings from images. The images corresponding to each visual word should be stored in a subdirectory of `imgs/`. If the subdirectory names are different from the words (e.g. if subdirectory names are WordNet IDs), a mapping between each word and its image subdirectory name should be provided in `data/word_dir_mapping.txt`. 

```bash
python3 create_visual_embeddings.py data/visual_embs.npz -layer fc1 -mapping data/word_dir_mapping.txt
```

The visual words and corresponding WordNet IDs used in Tang et al. are provided in `data/word_dir_mapping.txt`. The visual embeddings (extracted from layer fc1) used in Tang et al. are provided in `data/visual_embs.npz`. Unfortunately, due to copyright restrictions we are not able to share the original images used to create these embeddings.

2. Create the visually grounded embedding space. Given the visual embeddings of visual words, a sensory propagation method creates visually grounded embeddings of nonvisual  words. The amodal and grounded embedding spaces are represented by their covariance matrices and saved to the specified output file. 

```bash
python3 create_semantic_priors.py data/visual_embs.npz semantic_priors.npz
```

3. Create concreteness scores for weighting amodal and grounded embeddings. The [Brysbaert Concreteness Ratings](https://www.ncbi.nlm.nih.gov/pubmed/24142837) are provided in `data/conc_ratings.npz`. This function creates concreteness scores for each word in the embedding space by processing concreteness ratings and interpolating scores for missing words. 

```bash
python3 create_concreteness_scores.py semantic_priors.npz conc_scores.npy
```

4. Use grounded embedding spectrum. The GroundedSpectrum class loads the saved semantic priors and concreteness scores. 

```python
In [1]: from GroundedSpectrum import GroundedSpectrum
In [2]: gs = GroundedSpectrum('semantic_priors.npz', 'conc_scores.npy')
In [3]: gs.similarity('doctor', 'athlete', b = -10) # amodal similarity
Out[3]:
0.02106039500201583
In [4]: gs.similarity('doctor', 'athlete', b = 10) # visually grounded similarity
Out[4]:
0.4563762578004484
In [5]: gs.grounding_words('honor')[:10] # most strongly associated visual words under sensory propagation model
Out[5]:
[('memorial', 0.06103069404540094),
 ('soldier', 0.037841886270174624),
 ('uniform', 0.03660062783722907),
 ('cardinal', 0.03384936018858278),
 ('statue', 0.03382915745442363),
 ('candle', 0.030899885904507166),
 ('eagle', 0.03043478011620268),
 ('pilot', 0.029607891557204228),
 ('cathedral', 0.028945305232053853),
 ('heart', 0.028902362133141946)]
In [6]: gs.sigma(0)
Out[6]:
array([[ 1.        ,  0.69081572,  0.3095305 , ..., -0.24495695,
         0.24722239, -0.22175797],
       [ 0.69081572,  1.        ,  0.53911375, ..., -0.26954793,
         0.47671069, -0.24276763],
       [ 0.3095305 ,  0.53911375,  1.        , ..., -0.27715573,
         0.62269407,  0.01050194],
       ...,
       [-0.24495695, -0.26954793, -0.27715573, ...,  1.        ,
        -0.32578409, -0.32810192],
       [ 0.24722239,  0.47671069,  0.62269407, ..., -0.32578409,
         1.        , -0.04632105],
       [-0.22175797, -0.24276763,  0.01050194, ..., -0.32810192,
        -0.04632105,  1.        ]])
```
