
from keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
import time
import os.path
import itertools
import cv2
from glob import glob
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import math 
import keras


train_data_dir = 'Dataset/train'
valid_data_dir = 'Dataset/val'
batch_size=8
checkpointer = ModelCheckpoint(
    filepath=os.path.join('Dataset', 'checkpoints', 'idcVGG19.hdf5'),
    verbose=1,
    save_best_only=True)


early_stopper = EarlyStopping(monitor='val_loss', patience=10)

tensorboard = TensorBoard(log_dir=os.path.join('Dataset', 'logs'))
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('Dataset', 'logs', 'idcVGG19' + '-' + 'training-' + \
        str(timestamp) + '.log'))

def get_generators():
    train_datagen = ImageDataGenerator(
        featurewise_std_normalization=True,
        rescale=1./255,
                zoom_range=0.2,
                #brightness_range=(0.9, 1.1),
                rotation_range=0.2,
                #shear_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
                )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(50, 50),
        batch_size=batch_size,
        classes=['0', '1'],
        #classes=['Benign', 'Malignant'],
        #classes=['adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma'],
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(50, 50),
        batch_size=batch_size,
        classes=['0', '1'],
        #classes=['Benign', 'Malignant'],
        #classes=['adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma'],
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(weights='imagenet'):

    base_model = VGG19(weights=weights, include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    print("Number of layers in the base model: ", len(base_model.layers))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
def freeze_all_but_top(model):

    for layer in model.layers[15:]:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    for layer in model.layers:
        print(layer, layer.trainable)

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    from sklearn.utils import class_weight
    import numpy as np
  
    hist=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // batch_size,
        epochs=nb_epoch,
        #class_weight=class_weights,
        callbacks=callbacks)
    fig, axs = plt.subplots(1, 2, figsize = (15, 4))
    training_loss = hist.history['loss']
    validation_loss = hist.history['val_loss']
    training_accuracy = hist.history['accuracy']
    validation_accuracy = hist.history['val_accuracy']
    epoch_count = range(1, len(training_loss) + 1)
    #N=num_epochs
    axs[0].plot(epoch_count, training_loss, 'r--')
    axs[0].plot(epoch_count, validation_loss, 'b-')
    axs[0].legend(['Training Loss', 'Validation Loss'])
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[1].plot(epoch_count, training_accuracy, 'r--')
    axs[1].plot(epoch_count, validation_accuracy, 'b-')
    axs[1].legend(['Training Accuracy', 'Validation Accuracy'])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    fig.savefig('idcVGG19.png')
    return model

def main(weights_file):
    model = get_model()
    print(model.summary())

    print("Number of layers in the base model: ", len(model.layers))    
    generators = get_generators()


    if weights_file is None:
        print("Loading network from ImageNet weights.")

        model = freeze_all_but_top(model)
        model = train_model(model, 300, generators,
                        [checkpointer, early_stopper, tensorboard, csv_logger])
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    


if __name__ == '__main__':
    weights_file = None
    #print(len(data.classes))
    main(weights_file)
