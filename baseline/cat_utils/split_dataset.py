import os
import shutil
import random

def split_dataset(all_images_dir, train_dir, validation_dir, split_ratio=0.85):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    cat_breeds = os.listdir(all_images_dir)

    for breed in cat_breeds:
        breed_path = os.path.join(all_images_dir, breed)
        if not os.path.isdir(breed_path):
            continue
        
        images = os.listdir(breed_path)
        random.shuffle(images)

        train_size = int(len(images) * split_ratio)
        train_images = images[:train_size]
        validation_images = images[train_size:]

        train_breed_dir = os.path.join(train_dir, breed)
        validation_breed_dir = os.path.join(validation_dir, breed)

        if not os.path.exists(train_breed_dir):
            os.makedirs(train_breed_dir)
        if not os.path.exists(validation_breed_dir):
            os.makedirs(validation_breed_dir)

        for img in train_images:
            shutil.copy(os.path.join(breed_path, img), os.path.join(train_breed_dir, img))

        for img in validation_images:
            shutil.copy(os.path.join(breed_path, img), os.path.join(validation_breed_dir, img))

if __name__ == "__main__":
    all_images_dir = 'D:\\Shanghaitech\\NUS_SOC\\phase2\\project\\baseline\\cat_plus'
    train_dir = 'D:\\Shanghaitech\\NUS_SOC\\phase2\\project\\baseline\\split_data\\train'
    validation_dir = 'D:\\Shanghaitech\\NUS_SOC\\phase2\\project\\baseline\\split_data\\validation'
    
    split_dataset(all_images_dir, train_dir, validation_dir, split_ratio=0.85)
