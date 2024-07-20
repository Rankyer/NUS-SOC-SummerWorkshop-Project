from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import os
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

MODEL_NAME = 'cats_plus1.keras'
SAMPLE_PATH = './samples'

dict={0:'Pallas_cats', 1:'Persian_cat', 2:'Ragdoll', 3:'Singapura_cats', 4:'Sphynx_cats'}

# Takes in a loaded model, an image in numpy matrix format,
# And a label dictionary
def classify(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return (dict[themax], result[0][themax], themax)

# Load image
def load_image(image_fname):
    img = Image.open(image_fname)
    img = img.resize((249, 249))
    imgarray = np.array(img)/255.0
    final = np.expand_dims(imgarray, axis=0)
    return final

# Test main
def main():

    print("Loading model from ", MODEL_NAME)
    model = load_model(MODEL_NAME)
    print("Done")

    print("Now classifying files in ", SAMPLE_PATH)

    sample_files = listdir(SAMPLE_PATH)

    for filename in sample_files:
        filename = join(SAMPLE_PATH, filename)
        img = load_image(filename)
        label, prob, _ = classify(model, img)

        print("We think with certainty %3.2f that image %s is %s." % (prob, filename, label))

if __name__ == "__main__":
    main()
