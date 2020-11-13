# Semantic Embedding Spaces

This repository contains code for creating semantic embedding spaces used in the paper "Cortical Representations of Concrete and Abstract Concepts in Natural Language Combine Visual and Linguistic Representations" by Jerry Tang, Amanda I. Lebel, and Alexander G. Huth.  

## Usage

1. Download [data](https://utexas.box.com/shared/static/9s0kymmfpx5vc7lg8o2l47uxjim8bw2t.zip) and extract content into new `data/` directory. 

2. Extract CNN embeddings from images (if using custom images). The images corresponding to each visual word should be stored in a subdirectory of `imgs/`. If the subdirectory names are different from the words (e.g. if subdirectory names are WordNet IDs), a mapping between each word and its image subdirectory name should be provided in `data/word_dir_mapping.txt`. 

```bash
python3 create_visual_embeddings.py data/cnn_embs.npz -layer fc1 -mapping data/word_dir_mapping.txt
```

Alternatively, the visual words and corresponding WordNet IDs used in Tang et al. are provided in `data/word_dir_mapping.txt`, and the CNN embeddings (extracted from layer fc1 of VGG16) used in Tang et al. are provided in `data/visual_embs.npz`. Unfortunately, due to copyright restrictions we are not able to share the original images used to create these embeddings.

3. Create the linguistic and visual embedding spaces. Given the CNN embeddings of visual words, a sensory propagation method creates visually embeddings of visual and nonvisual words. The linguistic and visual embedding spaces are represented by their covariance matrices and saved to the specified output file. 

```bash
python3 create_priors.py data/cnn_embs.npz semantic_priors.npz
```

4. Create concreteness scores for weighting amodal and grounded embeddings. The [Brysbaert Concreteness Ratings](https://www.ncbi.nlm.nih.gov/pubmed/24142837) are provided in `data/conc_ratings.npz`. This function creates concreteness scores for each word in the embedding space by processing concreteness ratings and interpolating scores for missing words. 

```bash
python3 create_concreteness_scores.py semantic_priors.npz priors.npy
```

5. Create semantic embedding spaces by combining the visual and linguistic embedding spaces. The SemanticSpace class loads the saved priors and concreteness scores. 

```python
In [1]: from SemanticSpace import SemanticSpace
In [2]: ss = SemanticSpace('priors.npz', 'conc_scores.npy')
In [3]: ss.similarity('doctor', 'athlete', b = -10.0) # linguistic similarity
Out[3]:
0.02106039500201584
In [4]: ss.similarity('doctor', 'athlete', b = 10.0) # visual similarity
Out[4]:
0.4563762578004508
In [5]: ss.sigma(0)
Out[5]:
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
