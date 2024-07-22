import os
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

MODEL_NAME = 'cifar.keras'

def load_cifar10():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0
    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)
    return (train_x, train_y), (test_x, test_y)

def build_model(model_name):
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
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
        model.add(Dense(10, activation='softmax'))
    return model

def train(model, train_x, train_y, epochs, test_x, test_y, model_name):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    savemodel = ModelCheckpoint(model_name, save_best_only=True)
    stopmodel = EarlyStopping(min_delta=0.001, patience=10)

    print("Starting training.")

    model.fit(x=train_x, y=train_y, batch_size=32, validation_data=(test_x, test_y), shuffle=True, epochs=epochs, callbacks=[savemodel, stopmodel])

    print("Done. Now evaluating.")
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print("Test accuracy: %3.2f, loss: %3.2f" % (acc, loss))

(train_x, train_y), (test_x, test_y) = load_cifar10()
model = build_model(MODEL_NAME)
train(model, train_x, train_y, 50, test_x, test_y, MODEL_NAME)
