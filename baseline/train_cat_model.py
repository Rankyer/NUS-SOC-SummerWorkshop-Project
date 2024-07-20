from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import os.path
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

MODEL_FILE = "cats_plus.keras"

def create_model(num_hidden, num_classes):
    # We get the base model using InceptionV3 and the imagenet
    # weights that was trained on tens of thousands of images.
    base_model = InceptionV3(include_top = False, weights = 'imagenet')
    
    # Get the output layer, then does an average pooling of this
    # output, and feeds it to a final Dense layer that we
    # will train
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_hidden, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Set base model layers to be non trainable so we focus
    # our training only in the Dense layer. This lets us
    # adapt much faster and doesn't corrupt the weights that
    # were already trained on imagenet.
    for layer in base_model.layers:
        layer.trainable = False

    # Create a Functional Model (as opposed to the usual
    # Sequential Model that we create
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

def train(model_file, train_path, validation_path, num_hidden=200, num_classes=5, steps=32, num_epochs=20):
    if os.path.exists(model_file):
        print("\n*** Existing model found at %s. Loading...***\n" % model_file)
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n")
        model = create_model(num_hidden, num_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    checkpoint = ModelCheckpoint(model_file)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(249, 249),
        batch_size=5,
        class_mode='categorical'
    )
    
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(249, 249),
        batch_size=5,
        class_mode='categorical'
    )
    
    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=50
    )
    
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    
    # model.compile(optimizer=SGD(learning_rate=0.00001, momentum=0.9), loss='categorical_crossentropy')
    model.compile(optimizer=SGD(learning_rate=0.000000001, momentum=0.9), loss='categorical_crossentropy')
    
    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=50
    )


def main():
    train(MODEL_FILE, train_path="./split_data/train",
          validation_path="./split_data/validation", steps=120, num_epochs=20)

if __name__ == "__main__":
    main()

