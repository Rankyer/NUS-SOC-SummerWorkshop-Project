import os
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = 'food_recognition_model1.keras'
FOOD_SAMPLE_FOLDER = 'food_samples'

class_names = ["apple", "banana", "orange", "grape", "watermelon"]

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def load_and_predict_images(model, folder_path):
    for root, dirs, files in os.walk(folder_path):
        for img_name in files:
            img_path = os.path.join(root, img_name)
            if not os.path.isfile(img_path):
                continue
            img = preprocess_image(img_path)
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            print(f'Image: {img_name}, Predicted Class: {class_names[predicted_class]}')

def main():
    model = load_model(MODEL_PATH)
    load_and_predict_images(model, FOOD_SAMPLE_FOLDER)

if __name__ == "__main__":
    main()
