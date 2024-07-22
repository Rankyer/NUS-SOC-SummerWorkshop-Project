import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageOps, ImageFilter

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

MODEL_NAME = 'mnist-cnn.keras'
NUM_SAMPLE_FOLDER = './num_sample'


# def preprocess_image(img_path):
#     img = Image.open(img_path).convert('L')  # Convert to grayscale
#     img = img.filter(ImageFilter.MedianFilter())  # Apply a median filter to reduce noise
#     img = ImageOps.invert(img)  # Invert the image (MNIST is white on black)
#     img = ImageOps.autocontrast(img)  # Improve contrast
    
#     # Resize and center the image
#     img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
#     # Normalize the image
#     img = np.array(img)
#     img = img.astype('float32')
#     img /= 255.0  # Normalize to [0, 1]
    
#     # Add channel and batch dimensions
#     img = np.expand_dims(img, axis=-1)
#     img = np.expand_dims(img, axis=0)
#     return img

# def load_and_predict_images(model, folder_path):
#     for img_name in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_name)
#         img = preprocess_image(img_path)
#         prediction = model.predict(img)
#         predicted_class = np.argmax(prediction)
#         print(f'Image: {img_name}, Predicted Digit: {predicted_class}')

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28
    img = np.array(img)
    img = img.astype('float32')
    img /= 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def load_and_predict_images(model, folder_path):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = preprocess_image(img_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        print(f'Image: {img_name}, Predicted Digit: {predicted_class}')


def main():
    model = load_model(MODEL_NAME)
    load_and_predict_images(model, NUM_SAMPLE_FOLDER)

if __name__ == "__main__":
    main()


# without preprocessing
# def load_and_predict_images(model, folder_path):
#     for img_name in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_name)
#         img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
#         img = image.img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         img = img.astype('float32')
#         img /= 255.0
#         prediction = model.predict(img)
#         predicted_class = np.argmax(prediction)
#         print(f'Image: {img_name}, Predicted Digit: {predicted_class}')

# def main():
#     model = load_model(MODEL_NAME)
#     load_and_predict_images(model, NUM_SAMPLE_FOLDER)

# if __name__ == "__main__":
#     main()


# 
# def preprocess_image(img_path):
#     img = Image.open(img_path).convert('L')  # Convert to grayscale
#     img = img.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28
#     img = np.array(img)
#     img = img.astype('float32')
#     img /= 255.0  # Normalize to [0, 1]
#     img = np.expand_dims(img, axis=-1)  # Add channel dimension
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# def load_and_predict_images(model, folder_path):
#     for img_name in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_name)
#         img = preprocess_image(img_path)
#         prediction = model.predict(img)
#         predicted_class = np.argmax(prediction)
#         print(f'Image: {img_name}, Predicted Digit: {predicted_class}')
