import os
import random
import shutil

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

def save_sample_images(input_folder, output_folder, num_images=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    categories = os.listdir(input_folder)
    
    for category in categories:
        category_dir = os.path.join(input_folder, category)
        if not os.path.isdir(category_dir):
            continue
        
        images = os.listdir(category_dir)
        random.shuffle(images)
        sample_images = images[:num_images]
        
        category_output_dir = os.path.join(output_folder, category)
        os.makedirs(category_output_dir, exist_ok=True)
        
        for img in sample_images:
            src_path = os.path.join(category_dir, img)
            dst_path = os.path.join(category_output_dir, img)
            shutil.copy(src_path, dst_path)
    
    print(f"Saved sample images to {output_folder}")

if __name__ == "__main__":
    input_folder = './split_dataset1/train'
    output_folder = 'food_samples'
    save_sample_images(input_folder, output_folder)
