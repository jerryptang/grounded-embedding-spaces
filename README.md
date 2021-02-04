# Semantic Embedding Spaces

This repository contains code for creating semantic embedding spaces that combine visual and linguistic information, used in the paper "Cortical Representations of Concrete and Abstract Concepts in Language Combine Visual and Linguistic Representations" by Jerry Tang, Amanda I. Lebel, and Alexander G. Huth.  

## Usage

1. Download [data](https://utexas.box.com/shared/static/9s0kymmfpx5vc7lg8o2l47uxjim8bw2t.zip) and extract content into new `data/` directory. 

2. Extract CNN embeddings from images (if using custom images). The images corresponding to each visual word should be stored in a subdirectory of `imgs/`. If the subdirectory names are different from the words (e.g. if subdirectory names are WordNet IDs), a mapping between each word and its image subdirectory name should be provided in `data/word_dir_mapping.txt`. 

```bash
python3 create_cnn_embeddings.py data/cnn_embs.npz -layer fc1 -mapping data/word_dir_mapping.txt
```

Alternatively, the visual words and corresponding WordNet IDs used in Tang et al. are provided in `data/word_dir_mapping.txt`, and the CNN embeddings (extracted from layer fc1 of VGG16) used in Tang et al. are provided in `data/cnn_embs.npz`. Unfortunately, due to copyright restrictions we are not able to share the original images used to create these embeddings.

3. Create the linguistic and visual embedding spaces. Given the CNN embeddings of visual words, a sensory propagation method creates visual embeddings of visual and nonvisual words. The linguistic and visual embedding spaces are represented by their covariance matrices and saved to the specified output file. 

```bash
python3 create_priors.py data/cnn_embs.npz priors.npz
```

4. Create concreteness scores for weighting linguistic and visual embeddings. The [Brysbaert Concreteness Ratings](https://www.ncbi.nlm.nih.gov/pubmed/24142837) are provided in `data/conc_ratings.npz`. This function creates concreteness scores for each word in the embedding space by processing concreteness ratings and interpolating scores for missing words. 

```bash
python3 create_concreteness_scores.py priors.npz conc_scores.npy
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
In [5]: ss.sigma(0) # semantic similarities
Out[5]:
array([[ 1.        ,  0.68240295,  0.29996597, ..., -0.23954334,
         0.24379934, -0.22132871],
       [ 0.68240295,  1.        ,  0.52893927, ..., -0.27346699,
         0.46652609, -0.24104902],
       [ 0.29996597,  0.52893927,  1.        , ..., -0.29752363,
         0.64492018,  0.03358686],
       ...,
       [-0.23954334, -0.27346699, -0.29752363, ...,  1.        ,
        -0.33905557, -0.32810192],
       [ 0.24379934,  0.46652609,  0.64492018, ..., -0.33905557,
         1.        , -0.03228642],
       [-0.22132871, -0.24104902,  0.03358686, ..., -0.32810192,
        -0.03228642,  1.        ]])
```
