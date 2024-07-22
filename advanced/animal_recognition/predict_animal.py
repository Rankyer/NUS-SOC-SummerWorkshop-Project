import os
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = 'cifar.keras'
ANIMAL_SAMPLE_FOLDER = 'animal_sample'

# CIFAR-10 classes
class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def load_and_predict_images(model, folder_path):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = preprocess_image(img_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        print(f'Image: {img_name}, Predicted Class: {class_names[predicted_class]}')

def main():
    model = load_model(MODEL_PATH)
    load_and_predict_images(model, ANIMAL_SAMPLE_FOLDER)

if __name__ == "__main__":
    main()
