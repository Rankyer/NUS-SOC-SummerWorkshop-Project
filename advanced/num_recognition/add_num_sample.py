import os
import numpy as np
from keras.datasets import mnist
from PIL import Image
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)


def save_mnist_test_images(output_folder, num_images=10):
    # Load MNIST dataset
    (_, _), (test_x, test_y) = mnist.load_data()
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i in range(num_images):
        img_array = test_x[i]
        img_label = test_y[i]
        img = Image.fromarray(img_array)
        
        # Save image with label in the filename
        img.save(os.path.join(output_folder, f'mnist_{i}_label_{img_label}.png'))
    
    print(f"Saved {num_images} MNIST test images to {output_folder}")

if __name__ == "__main__":
    # output_folder = r'D:\Shanghaitech\NUS_SOC\phase2\NUS-Robotics-DL-project\advanced\num_recognition\num_sample'
    output_folder = './num_sample'
    save_mnist_test_images(output_folder, num_images=10)
