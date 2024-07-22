import os
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_NAME = 'food_recognition_model.keras'
DATASET_DIR = 'split_dataset'

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # 5 classes for 5 fruits
    return model

def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        DATASET_DIR + '/train',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR + '/val',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical'
    )

    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    savemodel = ModelCheckpoint(MODEL_NAME, save_best_only=True)
    stopmodel = EarlyStopping(min_delta=0.001, patience=10)

    model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[savemodel, stopmodel])

if __name__ == "__main__":
    train_model()
