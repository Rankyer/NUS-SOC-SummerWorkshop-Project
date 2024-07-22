from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os

def load_mnist():
    (train_x, train_y),(test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

    train_x=train_x.astype('float32')
    test_x = test_x.astype('float32')
    
    train_x /= 255.0
    test_x /= 255.0
        
    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)
        
    return (train_x, train_y), (test_x, test_y) 

MODEL_NAME = 'mnist-cnn.keras'

def buildmodel(model_name):
    if os.path.exists(model_name):
        model = load_model(model_name)                                                                                             
    else:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5,5),
        activation='relu',
        input_shape=(28, 28, 1), padding='same'))

        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
        model.add(Conv2D(128, kernel_size=(5,5), activation='relu'))
        model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='softmax'))

    return model

def train(model, train_x, train_y, epochs, test_x, test_y, model_name):

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.7), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    savemodel = ModelCheckpoint(model_name)
    stopmodel = EarlyStopping(min_delta=0.001, patience=10)

    print("Starting training.")

    model.fit(x=train_x, y=train_y, batch_size=32,
    validation_data=(test_x, test_y), shuffle=True,
    epochs=epochs, 
    callbacks=[savemodel, stopmodel])

    print("Done. Now evaluating.")
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print("Test accuracy: %3.2f, loss: %3.2f"%(acc, loss))

    (train_x, train_y),(test_x, test_y) = load_mnist()
    model = buildmodel(MODEL_NAME)
    train(model, train_x, train_y, 5, test_x, test_y, MODEL_NAME)
    