import os
import shutil
import random
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

def split_dataset(input_dir, output_dir, split_ratio=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        if not os.path.isdir(category_dir):
            continue
        
        images = os.listdir(category_dir)
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)

        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)

        for image in train_images:
            src_path = os.path.join(category_dir, image)
            dst_path = os.path.join(train_category_dir, image)
            shutil.copy(src_path, dst_path)

        for image in val_images:
            src_path = os.path.join(category_dir, image)
            dst_path = os.path.join(val_category_dir, image)
            shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    input_dir = "../dataset"
    output_dir = "../split_dataset"
    split_dataset(input_dir, output_dir)
