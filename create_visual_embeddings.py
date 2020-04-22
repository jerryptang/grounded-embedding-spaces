import os
import argparse
import numpy as np

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from config import IMG_DIR

def cnn_features(paths, model):
    word_imgs = []
    for path in paths:
        img = image.load_img(path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        word_imgs.append(img)
    return model.predict(np.vstack(word_imgs)).mean(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str)
    parser.add_argument('-layer', type=str, default='fc1')
    parser.add_argument('-mapping', type=str, default=None)
    args = parser.parse_args()
    
    # identify feature extraction layer from pretrained vgg16
    full_model = VGG16(weights="imagenet")
    feat_model = Model(inputs=full_model.input, outputs=full_model.get_layer(args.layer).output)

    # map words to image directories
    if args.mapping is None: 
        word_dir_mapping = dict(zip(os.listdir(IMG_DIR), os.listdir(IMG_DIR)))
    else:   
        with open(args.mapping, "r") as f:
            lines = f.readlines()
            word_dir_mapping = dict([l.strip('\n').split(' ') for l in lines])
        
    # extract visual embeddings for each word
    visual_embeddings = {}
    for word, word_dir in word_dir_mapping.items(): 
        if word_dir not in os.listdir(IMG_DIR): continue
        paths = []
        for img in os.listdir(os.path.join(IMG_DIR, word_dir)):
            paths.append(os.path.join(IMG_DIR, word_dir, img))
        visual_embeddings[word] = cnn_features(paths, feat_model)
    
    visual_words = sorted(list(visual_embeddings.keys()))
    visual_emb_arr = np.array([visual_embeddings[w] for w in visual_words])
    
    np.savez(args.outfile,
        embeddings=visual_emb_arr,
        visual_words=visual_words)        
