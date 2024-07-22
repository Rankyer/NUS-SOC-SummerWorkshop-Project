from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import KFold
import numpy as np
import os
import glob
from PIL import Image, UnidentifiedImageError
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息
os.environ['TERM'] = 'dumb'  # 禁用颜色输出


MODEL_FILE = "animals.keras"

def create_model(num_hidden, num_classes):
    base_model = InceptionV3(include_top=False, weights='imagenet')
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(0.5)(x)  # 添加Dropout层以防止过拟合
    predictions = Dense(num_classes, activation='softmax')(x)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def load_existing(model_file):
    model = load_model(model_file)
    numlayers = len(model.layers)
    for layer in model.layers[:numlayers-3]:
        layer.trainable = False

    for layer in model.layers[numlayers-3:]:
        layer.trainable = True
    
    return model

def train_cross_validation(model_file, data_path, num_hidden=200, num_classes=5, steps=32, num_epochs=20, k_folds=5):
    # Create data generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # List all image files
    image_files = glob.glob(os.path.join(data_path, '*/*'))

    # Filter out invalid image files
    image_files = [file for file in image_files if is_valid_image(file)]

    # Split data into K folds
    kfold = KFold(n_splits=k_folds, shuffle=True)

    fold_no = 1
    for train_index, val_index in kfold.split(image_files):
        print(f'\n*** Training on fold {fold_no} / {k_folds} ***\n')

        # Create new model or load existing one
        if os.path.exists(model_file):
            print(f"\n*** Existing model found at {model_file}. Loading...***\n")
            model = load_existing(model_file)
        else:
            print("\n*** Creating new model ***\n")
            model = create_model(num_hidden, num_classes)

        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='val_accuracy', mode='max')
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy', mode='max')
        reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, monitor='val_accuracy', mode='max')
        csv_logger = CSVLogger(f'training_log_fold_{fold_no}.csv')

        train_files = [image_files[i] for i in train_index]
        val_files = [image_files[i] for i in val_index]

        train_generator = datagen.flow_from_dataframe(
            dataframe=create_dataframe(train_files),
            x_col='filename',
            y_col='class',
            target_size=(249, 249),
            batch_size=5,
            class_mode='categorical'
        )

        validation_generator = datagen.flow_from_dataframe(
            dataframe=create_dataframe(val_files),
            x_col='filename',
            y_col='class',
            target_size=(249, 249),
            batch_size=5,
            class_mode='categorical'
        )

        model.fit(
            train_generator,
            steps_per_epoch=steps,
            epochs=num_epochs,
            callbacks=[checkpoint, early_stopping, reduce_lr, csv_logger],
            validation_data=validation_generator,
            validation_steps=50
        )

        fold_no += 1

def create_dataframe(file_list):
    import pandas as pd

    data = []
    for file_path in file_list:
        class_name = os.path.basename(os.path.dirname(file_path))
        data.append({'filename': file_path, 'class': class_name})

    return pd.DataFrame(data)

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # 验证图像文件
        return True
    except (UnidentifiedImageError, IOError, SyntaxError):
        print(f"Invalid image file detected and ignored: {file_path}")
        return False

def main():
    data_path = "E:\\file\\NUS\\animal_dataset"
    train_cross_validation(MODEL_FILE, data_path, num_hidden=200, num_classes=3, steps=120, num_epochs=30, k_folds=5)

if __name__ == "__main__":
    main()
