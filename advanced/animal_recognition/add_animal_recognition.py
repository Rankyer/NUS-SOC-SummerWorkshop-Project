import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from PIL import Image

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

def save_cifar10_images(output_folder, num_images=10):
    (_, _), (test_x, test_y) = cifar10.load_data()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i in range(num_images):
        img_array = test_x[i]
        img_label = test_y[i][0]
        img = Image.fromarray(np.uint8(img_array * 255))
        img.save(os.path.join(output_folder, f'cifar_{i}_label_{img_label}.png'))
    
    print(f"Saved {num_images} CIFAR-10 test images to {output_folder}")

if __name__ == "__main__":
    output_folder = 'animal_sample'
    save_cifar10_images(output_folder, num_images=10)
